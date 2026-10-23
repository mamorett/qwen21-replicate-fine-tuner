model: mamorett/qwen21-replicate-fine-tuner
test_model: mamorett/test-qwen21-fine-tuner

train:
  train_timeout: 900
  test_cases:

    # tiny training with autocaption
    - inputs:
        input_images: https://replicate.delivery/pbxt/LbXqvBzG3Bfo6CDkIkBNo8a8v6GWZGQrqxBrgeRpaQ1knkyD/me-tiny.zip
        steps: 10
        autocaption: true
        trigger_word: MY_CHAR
        hf_repo_id: mamorett/test-qwen21-lora-integration
        hf_token: "$HF_TOKEN"
      match_prompt: "A dictionary of a version and weights which are a .tar file"
