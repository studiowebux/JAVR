## DeepSpeech and Tensorflow Installation with Ansible

Installation and configuration with Ansible

- Deepspeech 0.5.1 & Tensorflow 1.13.1
- Deepspeech 0.7.0-alpha.3 & Tensorflow 1.15.0

[Complete Documentation (EN/FR)](https://studiowebux.github.io/deepspeech/)

## Definition

There is 10 scripts (including 1 optional)

1. The first one (`Install.yaml`) installs the required dependencies to execute the deployment.

2. The next 4 scripts, copy and install the dependencies to build and use deepspeech and tensorflow.

   1. `SOX.yaml`
   2. `KENLM.yaml`
   3. `SWIG.yaml`
   4. `BAZEL.yaml`

3. The sixth script (`DEEPSPEECH.yaml`) clone the _Mozilla/DeepSpeech_ repository, then checkout the branch to the chosen version and download the files with git-lfs (_approximately 1.7GB for version 0.5.1 & 700MB for version 0.7.0-alpha.3_)

4. The seventh script (`TENSORFLOW.yaml`) clone the _Mozilla/tensorflow_ repository, checkout the branch to the chosen version, then launch these commands to build tensorflow and deepspeech. This script calls an another one (`answer_tensorflow.yaml`) to answer automatically to the question asked during the configuration.

5. The eighth script (`TS_DEEPSPEECH.yaml`) build and install the files within `DeepSpeech/native_client/`,

   - _Javascript_
   - _ctcdecode_
   - _deepspeech_
   - _Python_

6. The last script (`Test.yaml`), it launches the demo included in the deepspeech directory. That test must complete without errors.

## Usage

### Commands

The SUDO password will be required (--ask-become)

#### DeepSpeech 0.5.1 and Tensorflow 1.13

```bash
ansible-playbook -i 1.2.3.4, Scripts/Main.yaml --extra-vars='{"tensorflow_version": "r1.13","DeepSpeech_version": "tags/v0.5.1","bazel_version": "0.19.1"}'--ask-become
```

#### DeepSpeech 0.7.0-alpha.3 and Tensorflow 1.15

```bash
ansible-playbook -i 1.2.3.4, Scripts/Main.yaml --ask-become
--extra-vars='{"tensorflow_version": "r1.15","DeepSpeech_version": "tags/v0.7.0-alpha.3","bazel_version": "0.24.1"}'
```

You can execute each command manually:

```bash
ansible-playbook -i 1.2.3.4, Scripts/Install.yaml --ask-become
ansible-playbook -i 1.2.3.4, Scripts/SOX.yaml --ask-become
ansible-playbook -i 1.2.3.4, Scripts/KENLM.yaml --ask-become --extra-vars '{"cores":8}'
ansible-playbook -i 1.2.3.4, Scripts/SWIG.yaml --ask-become
ansible-playbook -i 1.2.3.4, Scripts/BAZEL.yaml --extra-vars '{"bazel_version":"0.21.0"}' --ask-become
ansible-playbook -i 1.2.3.4, Scripts/DEEPSPEECH.yaml --extra-vars '{"DeepSpeech_version":"tags/v0.5.1"}' --ask-become
ansible-playbook -i 1.2.3.4, Scripts/TENSORFLOW.yaml --extra-vars '{"tensorflow_version":"r1.13"}' --ask-become
ansible-playbook -i 1.2.3.4, Scripts/TS_DEEPSPEECH.yaml --extra-vars '{"cores":8}' --ask-become
ansible-playbook -i 1.2.3.4, Scripts/Test.yaml --ask-become
```
