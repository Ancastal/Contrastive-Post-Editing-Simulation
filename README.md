# TODO

1. **Create a triplet for 10-50 segments.**
   - Test a functioning script for CPO.
   - Run the script and see if there are any issues.
   - Fix issues. Finalize the script.

**References**:  
- [Hugging Face TRL - CPO Trainer Documentation](https://huggingface.co/docs/trl/v0.8.6/cpo_trainer)  
- [CPO Example Script on GitHub](https://github.com/huggingface/trl/blob/main/examples/scripts/cpo.py)  

---

# Research Phase

- Investigate established methods for synthetic in-domain data generation.

---

# Generation Phase

1. Generate triplets from the original dataset.
2. Create a synthetic dataset using different ratios, e.g., 70-30. *(Look for references or prior examples.)*
3. Generate triplets for the synthetic data.

---

# Training and Evaluation Phase

1. Train the CPO model on the two datasets (original and original+synthetic).
2. Run inference on the datasets.
3. Evaluate results using metrics: BLEU, ChrF, TER, COMET-22
4. Obtain benchmark scores for comparison.

---

# CPO Dataset Structure

```python
cpo_dataset_dict = {
    "prompt": [
        "You are an experienced translator. Please translate 'cacio e pepe' from Italian into English",
    ],
    "chosen": [
        "Cheese and pepper",
    ],
    "rejected": [
        "Cacio and pepper",
    ],
}
