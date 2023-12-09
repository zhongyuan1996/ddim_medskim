import warnings
import torch
import tqdm
import numpy as np
class Evaluator:
    def __init__(self, model, device=None):
        self.model = model
        self.device = 'cpu' if device is None else device

    def lpl(self, logits, targets, lengths):
        ppl_list = []

        batch_size, num_visits, _ = logits.shape

        for i in range(batch_size):
            for visit in range(lengths[i]):
                # Extract logits and targets for the current visit of the current batch item
                logits_visit = logits[i, visit, :]
                target_visit = targets[i, visit, :]  # Extracting target codes for the specific visit

                prob = logits_visit.softmax(dim=-1)[target_visit == 1]
                nll = -torch.log(prob + 1e-10)
                ppl = nll.exp()

                if torch.isnan(ppl).any():
                    warnings.warn('NaN perplexity detected during lpl calculation')
                ppl_list.append(ppl)

        if ppl_list:
            median_ppl = torch.median(torch.cat(ppl_list))
            return median_ppl.item()
        else:
            print('No valid perplexity values found')
            return 0.0

    def eval(self, dataloader, pad_id, metric_list = []):

        if 'lpl' in metric_list:
            lpl_list = []

        for i, data in tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluating"):
            ehr, time_step, code_timegaps, code_mask, lengths, visit_timegaps, demo = data

            with torch.no_grad():
                if self.model.model == 'MedDiffGa':
                    logits, _, _, _ = self.model(ehr, None, lengths, time_step, code_mask, code_timegaps,
                                                                 visit_timegaps, demo)
                elif self.model.model == 'ehrGAN':
                    ehr_emb, _, _, mask, code_ehr = data
                    gen_ehr, real_ehr = self.model(ehr_emb, mask)

                multihot_ehr = torch.zeros_like(logits, dtype=torch.float32)

                for batch_idx in range(ehr.size(0)):
                    for seq_idx in range(ehr.size(1)):
                        for label in ehr[batch_idx, seq_idx]:
                            if label != pad_id:
                                multihot_ehr[batch_idx, seq_idx, label] = 1.0

                if 'lpl' in metric_list:
                    batch_lpl = self.lpl(logits, multihot_ehr, lengths)
                    lpl_list.append(batch_lpl)

        res = {}
        if 'lpl' in metric_list:
            res['lpl'] = np.median(lpl_list) if lpl_list else warnings.warn('No valid perplexity values found')

        return res