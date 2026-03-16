# Lab 8: Domain Adaptation Individual Report

## Student: Tony Nguyen

### 1. Description of Individual Contributions
- **PEFT Fine-Tuning**: Optimized `phi-2` model adaptation using QLoRA configurations, managing the training pipeline in Google Colab.
- **Dataset Engineering**: Scripted the `generate_dataset.py` pipeline to synthesize domain-specific instruction pairs from logistics and weather data.
- **Hardware Integration**: Tested and validated PEFT adapter loading on local GPU infrastructure to ensure consistency between training and inference environments.
- **Prompt Optimization**: Iterated on instruction formats to improve grounding and reduce hallucinations in the adapted model's responses.

### 2. Contribution Percentage
**33.3%**

### 3. Repository Evidence
* **Commit `e62c8de`:** Uploaded adapted model weights and PEFT configurations.
* **Commit `ee06d4d`:** Integrated synthetic domain-adapted datasets into the project structure.
* **Commit `d233c01`:** Initial framework setup for Lab 8 adaptation methods and notebook environments.
* **Commit `cf279e3`:** Updated reports with logistics instruction logic and domain-adapted reasoning patterns.

### 4. AI Tools Used
We utilized the **Google Antigravity AI Agent** operating on a local windows environment and **Claude Code** enhancing the quality of prompts and debugging process.
Specifically, it was used to:
1. Generate the synthetic dataset and the structure of `evaluation_queries.json` including the metamorphic test pairs.
2. Draft the implementation code for Python processing scripts, particularly `peft_finetuning.py`, `demo_dashboard.py`, and `evaluation.py`.
3. Provide automated troubleshooting and trace-log interpretations when resolving environment issues such as the `SFTTrainer` class inheritance inconsistencies in newer TRL dependencies on Colab.
4. Iterate and update evaluation pipelines on the local GPU machine.
