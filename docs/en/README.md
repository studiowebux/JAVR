# Introduction

# Goals

Deploy Deepspeech and Tensorflow on Ubuntu 18.04 without AVX and/or SSE support.  
Also, the server uses only the CPU.

# Requirements

- A server with Ubuntu 18.04 installed (VM or physical)
- Minimum of 12GB of RAM
- Minimum of 8 cores (The tests have been done with : Intel(R) Xeon(R) CPU L5520 @ 2.27GHz)

> The physical server is a "HP ProLiant DL160 G6" with oVirt as hypervisor.  
> -- The VM used for testing has 16GB of RAM and 8 cores with an SSD of 30GB.

# Scripts

Available on Github : <a href="https://github.com/studiowebux/deepspeech" target="_blank" rel="noopener noreferrer">Github Deepspeech</a>

# Installation

# Goal

Prepare the environment to deploy Deepspeech and Tensorflow on Ubuntu 18.04

# Step 1 - Define which version to deploy

| Mozilla/Tensorflow Version | Mozilla/DeepSpeech Version | Bazel Version | Tesed | YAML         |
| -------------------------- | -------------------------- | ------------- | ----- | ------------ |
| r1.13                      | tags/v0.5.1                | 0.19.1        | YES   | Main.yaml    |
| r1.15                      | tags/v0.7.0-alpha.3        | 0.24.1        | YES   | Main_v2.yaml |

> NOTE, only these versions have been tested.

based on the chosen version, you must have these dependencies

> They are available directly in the github

## DeepSpeech V0.5.1 + Tensorflow R1.13

- bazel_0.19.1-linux-x86_64.deb
- sox-14.4.2.tar
- swig-4.0.0.tar
- kenlm.tar

## DeepSpeech V0.7.0-alpha.3 + Tensorflow R1.15

- bazel_0.24.1-linux-x86_64.deb
- sox-14.4.2.tar
- swig-4.0.0.tar
- kenlm.tar

## Bazel

To know which versions are compatible with the tensorflow version,

For example with _Mozilla/Tensorflow R1.13_

```bash
cat /srv/tensorflow/configure.py | grep check_bazel_version
def check_bazel_version(min_version, max_version):
check_bazel_version('0.19.0', '0.21.0')
```

At the bottom of this page, you can also find the information : <a href="https://www.tensorflow.org/install/source" target="_blank" rel="noopener noreferrer">Tensorflow Build From Source</a>

# Step 2 - Install tools on the deployment machine (ansible controller)

## Ansible

Ansible is used to deploy the whole stack,

The official documentation : <a href="https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html" target="_blank" rel="noopener noreferrer">Ansible Installation</a>

To check if the installation has been done successfully, launch these commands :

```bash
ansible --version
ansible-playbook --version
```

## Git

Official link to install git on your system : <a href="https://git-scm.com" target="_blank" rel="noopener noreferrer">Git Installation</a>

To check if the installation has been done successfully, launch this command :

```bash
git --version
```

## SSH Key

This is recommanded to use an SSH key to establish the session.

To generate an SSH key launch this command and follow the instructions on the console,

```bash
ssh-keygen
```

To copy your public key to the server, launch this command :

```bash
ssh-copy-id user@1.2.3.4
```

> Replace 'user' and '1.2.3.4' for your server information

using the SSH key, ansible will be able to connect automatically to the server using the private key stored on the controller.

# Summary

- Your deployment machine is configured with all the tools.
- You have all dependencies to launch the deployment script based on the chosen version.

# Deployment

# Goal

Launch the script named "Main.yaml" to deploy the whole stack on your server.

# Commands

The SUDO password will be required (`--ask-become`)

## DeepSpeech 0.5.1 and Tensorflow 1.13 :

```bash
ansible-playbook -i 1.2.3.4, Scripts/Main.yaml --extra-vars='{"tensorflow_version": "r1.13","DeepSpeech_version": "tags/v0.5.1","bazel_version": "0.19.1"}'--ask-become
```

## DeepSpeech 0.7.0-alpha.3 and Tensorflow 1.15 :

```bash
ansible-playbook -i 1.2.3.4, Scripts/Main.yaml --ask-become
--extra-vars='{"tensorflow_version": "r1.15","DeepSpeech_version": "tags/v0.7.0-alpha.3","bazel_version": "0.24.1"}'
```

# Scripts definition

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

To install the tensorflow version available online,

> The recommended way is to use the version compiled during the deployment.

```bash
ansible-playbook -i 1.2.3.4, Scripts/FIX_TENSORFLOW.yaml --extra-vars '{"tensorflow_version":"r1.13"}' --ask-become
```

# Notes

- The **Build Tensorflow** step takes a lot of time, (if possible, add more CPU to speed up the process, _for example, with 12 cores/threads at 2.2GHzm it takes approximateely 1H15_)
- To see the progression or to know the system state, you can open an SSH session and launch some commands, like `top` or `ps -aec` to see what is going on. It is also possible to print the output of the process with this command, `cat /proc/[process ID of bazel-real]/fd/2`

# Bundle

# DeepSpeech v0.5.1 & Mozilla/TensorFlow v1.13.1

```bash
tgingras@javr:~\$ deepspeech
Usage: deepspeech --model MODEL --alphabet ALPHABET [--lm LM --trie TRIE] --audio AUDIO [-t][-e]

Running DeepSpeech inference.

    --model MODEL		Path to the model (protocol buffer binary file)
    --alphabet ALPHABET	Path to the configuration file specifying the alphabet used by the network
    --lm LM			Path to the language model binary file
    --trie TRIE		Path to the language model trie file created with native_client/generate_trie
    --audio AUDIO		Path to the audio file to run (WAV format)
    -t			Run in benchmark mode, output mfcc & inference time
    --extended		Output string from extended metadata
    --json			Extended output, shows word timings as JSON
    --stream size		Run in stream mode, output intermediate results
    --help			Show help
    --version		Print version and exits

TensorFlow: v1.13.1-13-g174b4760eb
DeepSpeech: v0.5.1-0-g4b29b788
```

# DeepSpeech v0.7.0-alpha.3 & Mozilla/TensorFlow v1.15.0

```bash
root@javr:/srv/DeepSpeech# deepspeech --version
DeepSpeech 0.7.0-alpha.3
```

# Issues

# Issue #1 - Tensorflow built with AVX

> The system doesn't support AVX neither SSE

## Solution #1

Use the version built by the script `TENSORFLOW.yaml`

To know which version of bazel to use,

```bash
cat /srv/tensorflow/configure.py | grep check_bazel_version
```

## Solution #2

Use, if available, a version built by the community, <a href="https://github.com/yaroslavvb/tensorflow-community-wheels/issues" target="_blank" rel="noopener noreferrer">Github</a>

This solution is scripted in this file : `FIX_TENSORFLOW.yaml`

<a href="https://github.com/Tzeny/tensorflowbuilds/raw/master/tensorflow-1.13.1-cp36-cp36m-linux_x86_64.whl" target="_blank" rel="noopener noreferrer">tensorflow-1.13.1-cp36-cp36m-linux_x86_64.whl</a> -
<a href="https://github.com/yaroslavvb/tensorflow-community-wheels/issues/111" target="_blank" rel="noopener noreferrer">Source</a>

To install the package,

```bash
wget https://github.com/Tzeny/tensorflowbuilds/raw/master/tensorflow-1.13.1-cp36-cp36m-linux_x86_64.whl -O /tmp/tensorflow-1.13.1-cp36-cp36m-linux_x86_64.whl
pip3 uninstall tensorflow
pip3 install /tmp/tensorflow-1.13.1-cp36-cp36m-linux_x86_64.whl
```

# Error

```bash
root@javr:/srv/DeepSpeech# ./bin/run-ldc93s1.sh

- [ ! -f DeepSpeech.py ]
- [ ! -f data/ldc93s1/ldc93s1.csv ]
- echo Downloading and preprocessing LDC93S1 example data, saving in ./data/ldc93s1.
  Downloading and preprocessing LDC93S1 example data, saving in ./data/ldc93s1.
- python -u bin/import_ldc93s1.py ./data/ldc93s1
  No path "./data/ldc93s1" - creating ...
  No archive "./data/ldc93s1/LDC93S1.wav" - downloading...
  Progress | | N/A% completedNo archive "./data/ldc93s1/LDC93S1.txt" - downloading...
  Progress |#######################################################################################################| 100% completed
  Progress |#######################################################################################################| 100% completed
- [ -d ]
- python -c from xdg import BaseDirectory as xdg; print(xdg.save_data_path("deepspeech/ldc93s1"))
- checkpoint_dir=/root/.local/share/deepspeech/ldc93s1
- export CUDA_VISIBLE_DEVICES=0
- python -u DeepSpeech.py --noshow_progressbar --train_files data/ldc93s1/ldc93s1.csv --test_files data/ldc93s1/ldc93s1.csv --train_batch_size 1 --test_batch_size 1 --n_hidden 100 --epochs 200 --checkpoint_dir /root/.local/share/deepspeech/ldc93s1
  2020-04-19 15:00:35.785929: F tensorflow/core/platform/cpu_feature_guard.cc:37] The TensorFlow library was compiled to use AVX instructions, but these aren't available on your machine.
  Aborted (core dumped)
```

# Issue #2 - Numpy returns an error

The numpy version installed is too old, simply upgrade numpy to resolve this issue

```bash
pip3 install --upgrade numpy
```

```bash
root@javr:/srv/DeepSpeech# pip3 install --upgrade numpy
Collecting numpy
Downloading numpy-1.18.2-cp36-cp36m-manylinux1_x86_64.whl (20.2 MB)
|████████████████████████████████| 20.2 MB 2.8 MB/s
Installing collected packages: numpy
Attempting uninstall: numpy
Found existing installation: numpy 1.15.4
Uninstalling numpy-1.15.4:
Successfully uninstalled numpy-1.15.4
Successfully installed numpy-1.18.2
```

## Error

```bash
root@javr:/srv/DeepSpeech# ./bin/run-ldc93s1.sh

- [ ! -f DeepSpeech.py ]
- [ ! -f data/ldc93s1/ldc93s1.csv ]
- [ -d ]
- python -c from xdg import BaseDirectory as xdg; print(xdg.save_data_path("deepspeech/ldc93s1"))
- checkpoint_dir=/root/.local/share/deepspeech/ldc93s1
- export CUDA_VISIBLE_DEVICES=0
- python -u DeepSpeech.py --noshow_progressbar --train_files data/ldc93s1/ldc93s1.csv --test_files data/ldc93s1/ldc93s1.csv --train_batch_size 1 --test_batch_size 1 --n_hidden 100 --epochs 200 --checkpoint_dir /root/.local/share/deepspeech/ldc93s1
  ModuleNotFoundError: No module named 'numpy.core.\_multiarray_umath'
  ImportError: numpy.core.multiarray failed to import

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
File "<frozen importlib._bootstrap>", line 968, in \_find_and_load
SystemError: <class '\_frozen_importlib.\_ModuleLockManager'> returned a result with an error set
ImportError: numpy.core.\_multiarray_umath failed to import
ImportError: numpy.core.umath failed to import
2020-04-19 15:04:27.224250: F tensorflow/python/lib/core/bfloat16.cc:675] Check failed: PyBfloat16_Type.tp_base != nullptr
Aborted (core dumped)
```

# Issue #3 - No `ds_ctcdecoder`

This module must be built, this task is done in `TS_DEEPSPEECH.yaml`

> This error occured, because the script was run manually, and the script hasn't been executed at the moment of the error.

To install the module manually,

```bash
cd /srv/DeepSpeech/native_client/ctcdecode
make bindings NUM_PROCESSES=8
pip3 install dist/\*.whl
```

# Error

```bash
root@javr:/srv/DeepSpeech# ./bin/run-ldc93s1.sh

- [ ! -f DeepSpeech.py ]
- [ ! -f data/ldc93s1/ldc93s1.csv ]
- [ -d ]
- python -c from xdg import BaseDirectory as xdg; print(xdg.save_data_path("deepspeech/ldc93s1"))
- checkpoint_dir=/root/.local/share/deepspeech/ldc93s1
- export CUDA_VISIBLE_DEVICES=0
- python -u DeepSpeech.py --noshow_progressbar --train_files data/ldc93s1/ldc93s1.csv --test_files data/ldc93s1/ldc93s1.csv --train_batch_size 1 --test_batch_size 1 --n_hidden 100 --epochs 200 --checkpoint_dir /root/.local/share/deepspeech/ldc93s1
  /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/dtypes.py:526: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  \_np_qint8 = np.dtype([("qint8", np.int8, 1)])
  /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/dtypes.py:527: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  \_np_quint8 = np.dtype([("quint8", np.uint8, 1)])
  /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/dtypes.py:528: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  \_np_qint16 = np.dtype([("qint16", np.int16, 1)])
  /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/dtypes.py:529: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  \_np_quint16 = np.dtype([("quint16", np.uint16, 1)])
  /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/dtypes.py:530: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  \_np_qint32 = np.dtype([("qint32", np.int32, 1)])
  /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/dtypes.py:535: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  np_resource = np.dtype([("resource", np.ubyte, 1)])
  Traceback (most recent call last):
  File "DeepSpeech.py", line 18, in <module>
  from ds_ctcdecoder import ctc_beam_search_decoder, Scorer
  ModuleNotFoundError: No module named 'ds_ctcdecoder'
```

# Issue #4 - kenlm returns an error

This error is caused by `git-lfs`. The installation of that package wasn't done properly, then the LFS files were not downloaded.

> This error occured during the manual test.

## Solution #1

Launch the script corectly, the LFS step is included in the script : `DEEPSPEECH.yaml`

## Solution #2

Launch these commands manually :

> Update the version to the one you use

```bash
git clone https://github.com/mozilla/DeepSpeech.git
cd /srv/DeepSpeech
git checkout tags/v0.5.1
git-lfs pull
```

After the download, the test is PASS

```bash
...
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/training/saver.py:1266: checkpoint_exists (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.
Instructions for updating:
Use standard file APIs to check for files with this prefix.
I Restored variables from most recent checkpoint at /root/.local/share/deepspeech/ldc93s1/train-200, step 200
Testing model on data/ldc93s1/ldc93s1.csv
I Test epoch...
Test on data/ldc93s1/ldc93s1.csv - WER: 0.000000, CER: 0.000000, loss: 4.539505

---

WER: 0.000000, CER: 0.000000, loss: 4.539505

- src: "she had your dark suit in greasy wash water all year"
- res: "she had your dark suit in greasy wash water all year"

---

```

# Error

```bash
root@javr:/srv/DeepSpeech# ./bin/run-ldc93s1.sh

- [ ! -f DeepSpeech.py ]
- [ ! -f data/ldc93s1/ldc93s1.csv ]
- [ -d ]
- python -c from xdg import BaseDirectory as xdg; print(xdg.save_data_path("deepspeech/ldc93s1"))
- checkpoint_dir=/root/.local/share/deepspeech/ldc93s1
- export CUDA_VISIBLE_DEVICES=0
- python -u DeepSpeech.py --noshow_progressbar --train_files data/ldc93s1/ldc93s1.csv --test_files data/ldc93s1/ldc93s1.csv --train_batch_size 1 --test_batch_size 1 --n_hidden 100 --epochs 200 --checkpoint_dir /root/.local/share/deepspeech/ldc93s1
  /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/dtypes.py:526: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  \_np_qint8 = np.dtype([("qint8", np.int8, 1)])
  /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/dtypes.py:527: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  \_np_quint8 = np.dtype([("quint8", np.uint8, 1)])
  /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/dtypes.py:528: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  \_np_qint16 = np.dtype([("qint16", np.int16, 1)])
  /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/dtypes.py:529: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  \_np_quint16 = np.dtype([("quint16", np.uint16, 1)])
  /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/dtypes.py:530: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  \_np_qint32 = np.dtype([("qint32", np.int32, 1)])
  /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/dtypes.py:535: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  np_resource = np.dtype([("resource", np.ubyte, 1)])
  WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/data/ops/dataset_ops.py:429: py_func (from tensorflow.python.ops.script_ops) is deprecated and will be removed in a future version.
  Instructions for updating:
  tf.py_func is deprecated in TF V2. Instead, use
  tf.py_function, which takes a python function which manipulates tf eager
  tensors instead of numpy arrays. It's easy to convert a tf eager tensor to
  an ndarray (just call tensor.numpy()) but having access to eager tensors
  means `tf.py_function`s can use accelerators such as GPUs as well as
  being differentiable using a gradient tape.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/data/ops/iterator_ops.py:358: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/contrib/rnn/python/ops/lstm_ops.py:696: to_int64 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.cast instead.
I Initializing variables...
I STARTING Optimization
I Training epoch 0...
I Finished training epoch 0 - loss: 389.854401
I Training epoch 1...
I Finished training epoch 1 - loss: 351.508087
I Training epoch 2...
I Finished training epoch 2 - loss: 330.364197
I Training epoch 3...
I Finished training epoch 3 - loss: 308.195038
I Training epoch 4...
I Finished training epoch 4 - loss: 283.456207
I Training epoch 5...
I Finished training epoch 5 - loss: 261.468903
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/training/saver.py:966: remove_checkpoint (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.
Instructions for updating:
Use standard file APIs to delete files with this prefix.
I Training epoch 6...
I Finished training epoch 6 - loss: 245.990280
I Training epoch 7...
...
I Training epoch 199...
I Finished training epoch 199 - loss: 10.046951
I FINISHED optimization in 0:00:41.101047
Loading the LM will be faster if you build a binary file.
Reading data/lm/lm.binary
----5---10---15---20---25---30---35---40---45---50---55---60---65---70---75---80---85---90---95--100
terminate called after throwing an instance of 'lm::FormatLoadException'
what(): ../kenlm/lm/read_arpa.cc:65 in void lm::ReadARPACounts(util::FilePiece&, std::vector<long unsigned int>&) threw FormatLoadException.
first non-empty line was "version https://git-lfs.github.com/spec/v1" not \data\. Byte: 43
Aborted (core dumped)
```

# Issue #5 - Tensorflow built locally

> Attempt #1 and #5 are PASS

## Attempt #1 (PASS)

This version is compatible : <a href="https://github.com/yaroslavvb/tensorflow-community-wheels/issues/111" target="_blank" rel="noopener noreferrer">Github</a>

> But the goal is to build tensorflow from source

## Attempt #2 (FAIL)

_bazel_ version : **0.21.0**  
_tensorflow_ version : **Mozilla/tensoflow** _checkout_ : **r1.13**

Launch this command:

```bash
bazel build --config=monolithic -c opt --copt=-O3 --copt="-D_GLIBCXX_USE_CXX11_ABI=0" --copt=-fvisibility=hidden //tensorflow/tools/pip_package:build_pip_package //native_client:libdeepspeech.so //native_client:generate_trie
```

## Attempt #3 (FAIL)

Using the "default" version

```bash
pip3 install tensorflow==1.13.1
```

### result

```bash
tgingras@javr:/srv/DeepSpeech\$ ./DeepSpeech.py
2020-04-20 01:19:24.380911: F tensorflow/core/platform/cpu_feature_guard.cc:37] The TensorFlow library was compiled to use AVX instructions, but these aren't available on your machine.
Aborted (core dumped)
```

## Attempt #4 (FAIL)

Use bazel version 0.25.2
then launch the command below,

> Error returned : the bazel version must be between 0.19.0 and 0.21.0 to build Mozilla/Tensorflow r1.13

```bash
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
```

## Attempt #5 (PASS)

Using bazel version 0.19.1, the built and package install were successful.

Link for bazel 0.19.1 : <a href="https://github.com/bazelbuild/bazel/releases/download/0.19.1/bazel_0.19.1-linux-x86_64.deb" target="_blank" rel="noopener noreferrer">Github</a>

Launch these commands:

```bash
cd /srv
wget https://github.com/bazelbuild/bazel/releases/download/0.19.1/bazel_0.19.1-linux-x86_64.deb
dpkg -i bazel_0.19.1-linux-x86_64.deb
apt install openjdk-8-jdk
apt --fix-broken install
cd tensorflow/
./configure
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/pkg
pip3 uninstall -y tensorflow
pip3 install /tmp/pkg/tensorflow-1.13.1-cp36-cp36m-linux_x86_64.whl
cd ..
cd DeepSpeech/
./DeepSpeech.py
./bin/run-ldc93s1.sh
```

> the `apt` commands are required, because there was a conflict with java.

### Bundle

#### Tensorflow Version

```bash
root@javr:/srv/tensorflow# git status
On branch r1.13
Your branch is up to date with 'origin/r1.13'.
```

#### Bazel version

```bash
root@javr:/srv/DeepSpeech# bazel version
WARNING: --batch mode is deprecated. Please instead explicitly shut down your Bazel server using the command "bazel shutdown".
Build label: 0.19.1
Build target: bazel-out/k8-opt/bin/src/main/java/com/google/devtools/build/lib/bazel/BazelServer_deploy.jar
Build time: Mon Nov 12 15:32:30 2018 (1542036750)
Build timestamp: 1542036750
Build timestamp as int: 1542036750
```

#### Compilation et installation de tensorflow 1.13.`

```bash
[8,451 / 8,453] Compiling tensorflow/core/kernels/conv_ops_fused.cc; 118s locTarget //tensorflow/tools/pip_package:build_pip_package up-to-date:
bazel-bin/tensorflow/tools/pip_package/build_pip_package
INFO: Elapsed time: 3997.612s, Critical Path: 337.19s, Remote (0.00% of the time): [queue: 0.00%, setup: 0.00%, process: 0.00%]
INFO: 7722 processes: 7722 local.
INFO: Build completed successfully, 8453 total actions
root@javr:/srv/tensorflow# ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/pkg
Mon Apr 20 11:35:15 UTC 2020 : === Preparing sources in dir: /tmp/tmp.P1f2NhAcXO
/srv/tensorflow /srv/tensorflow
/srv/tensorflow
Mon Apr 20 11:35:27 UTC 2020 : === Building wheel
warning: no files found matching '_.pyd' under directory '_'
warning: no files found matching '_.pd' under directory '_'
warning: no files found matching '_.dll' under directory '_'
warning: no files found matching '_.lib' under directory '_'
warning: no files found matching '_.h' under directory 'tensorflow/include/tensorflow'
warning: no files found matching '_' under directory 'tensorflow/include/Eigen'
warning: no files found matching '_.h' under directory 'tensorflow/include/google'warning: no files found matching '_' under directory 'tensorflow/include/third_party'
warning: no files found matching '\*' under directory 'tensorflow/include/unsupported'
Mon Apr 20 11:35:53 UTC 2020 : === Output wheel file is in: /tmp/pkg
root@javr:/srv/tensorflow# pip3 uninstall -y tensorflow
Found existing installation: tensorflow 1.13.1
Uninstalling tensorflow-1.13.1:
Successfully uninstalled tensorflow-1.13.1
root@javr:/srv/tensorflow# pip3 install /tmp/p
pip-uninstall-dxafw9ii/ pkg/
root@javr:/srv/tensorflow# pip3 install /tmp/pkg/tensorflow-1.13.1-cp36-cp36m-linux_x86_64.whl
Processing /tmp/pkg/tensorflow-1.13.1-cp36-cp36m-linux_x86_64.whl
Requirement already satisfied: six>=1.10.0 in /usr/lib/python3/dist-packages (from tensorflow==1.13.1) (1.11.0)
Requirement already satisfied: gast>=0.2.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow==1.13.1) (0.3.3)
Requirement already satisfied: keras-applications>=1.0.6 in /usr/local/lib/python3.6/dist-packages (from tensorflow==1.13.1) (1.0.8)
Requirement already satisfied: absl-py>=0.1.6 in /usr/local/lib/python3.6/dist-packages (from tensorflow==1.13.1) (0.9.0)
Requirement already satisfied: grpcio>=1.8.6 in /usr/local/lib/python3.6/dist-packages (from tensorflow==1.13.1) (1.28.1)
Requirement already satisfied: tensorboard<1.14.0,>=1.13.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow==1.13.1) (1.13.1)
Requirement already satisfied: termcolor>=1.1.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow==1.13.1) (1.1.0)
Requirement already satisfied: keras-preprocessing>=1.0.5 in /usr/local/lib/python3.6/dist-packages (from tensorflow==1.13.1) (1.1.0)
Requirement already satisfied: astor>=0.6.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow==1.13.1) (0.8.1)
Requirement already satisfied: numpy>=1.13.3 in /usr/local/lib/python3.6/dist-packages (from tensorflow==1.13.1) (1.18.3)
Requirement already satisfied: tensorflow-estimator<1.14.0rc0,>=1.13.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow==1.13.1) (1.13.0)
Requirement already satisfied: wheel>=0.26 in /usr/local/lib/python3.6/dist-packages (from tensorflow==1.13.1) (0.31.0)
Requirement already satisfied: protobuf>=3.6.1 in /usr/local/lib/python3.6/dist-packages (from tensorflow==1.13.1) (3.11.3)
Requirement already satisfied: h5py in /usr/local/lib/python3.6/dist-packages (from keras-applications>=1.0.6->tensorflow==1.13.1) (2.10.0)
Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.6/dist-packages (from tensorboard<1.14.0,>=1.13.0->tensorflow==1.13.1) (3.2.1)
Requirement already satisfied: werkzeug>=0.11.15 in /usr/local/lib/python3.6/dist-packages (from tensorboard<1.14.0,>=1.13.0->tensorflow==1.13.1) (1.0.1)
Requirement already satisfied: mock>=2.0.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow-estimator<1.14.0rc0,>=1.13.0->tensorflow==1.13.1) (4.0.2)
Requirement already satisfied: setuptools in /usr/local/lib/python3.6/dist-packages (from protobuf>=3.6.1->tensorflow==1.13.1) (39.1.0)
Installing collected packages: tensorflow
Successfully installed tensorflow-1.13.1
```

#### Launch the demo

```bash
root@javr:/srv/tensorflow# cd ..
root@javr:/srv# cd DeepSpeech/
root@javr:/srv/DeepSpeech# ./DeepSpeech.py
/usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/dtypes.py:526: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
\_np_qint8 = np.dtype([("qint8", np.int8, 1)])
/usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/dtypes.py:527: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
\_np_quint8 = np.dtype([("quint8", np.uint8, 1)])
/usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/dtypes.py:528: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
\_np_qint16 = np.dtype([("qint16", np.int16, 1)])
/usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/dtypes.py:529: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
\_np_quint16 = np.dtype([("quint16", np.uint16, 1)])
/usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/dtypes.py:530: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
\_np_qint32 = np.dtype([("qint32", np.int32, 1)])
/usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/dtypes.py:535: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
np_resource = np.dtype([("resource", np.ubyte, 1)])
root@javr:/srv/DeepSpeech# ./bin/run-ldc93s1.sh

- [ ! -f DeepSpeech.py ]
- [ ! -f data/ldc93s1/ldc93s1.csv ]
- [ -d ]
- python -c from xdg import BaseDirectory as xdg; print(xdg.save_data_path("deepspeech/ldc93s1"))
- checkpoint_dir=/root/.local/share/deepspeech/ldc93s1
- export CUDA_VISIBLE_DEVICES=0
- python -u DeepSpeech.py --noshow_progressbar --train_files data/ldc93s1/ldc93s1.csv --test_files data/ldc93s1/ldc93s1.csv --train_batch_size 1 --test_batch_size 1 --n_hidden 100 --epochs 200 --checkpoint_dir /root/.local/share/deepspeech/ldc93s1
  /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/dtypes.py:526: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  \_np_qint8 = np.dtype([("qint8", np.int8, 1)])
  /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/dtypes.py:527: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  \_np_quint8 = np.dtype([("quint8", np.uint8, 1)])
  /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/dtypes.py:528: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  \_np_qint16 = np.dtype([("qint16", np.int16, 1)])
  /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/dtypes.py:529: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  \_np_quint16 = np.dtype([("quint16", np.uint16, 1)])
  /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/dtypes.py:530: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  \_np_qint32 = np.dtype([("qint32", np.int32, 1)])
  /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/dtypes.py:535: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  np_resource = np.dtype([("resource", np.ubyte, 1)])
  WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/data/ops/dataset_ops.py:429: py_func (from tensorflow.python.ops.script_ops) is deprecated and will be removed in a future version.
  Instructions for updating:
  tf.py_func is deprecated in TF V2. Instead, use
  tf.py_function, which takes a python function which manipulates tf eager
  tensors instead of numpy arrays. It's easy to convert a tf eager tensor to
  an ndarray (just call tensor.numpy()) but having access to eager tensors
  means `tf.py_function`s can use accelerators such as GPUs as well as
  being differentiable using a gradient tape.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/data/ops/iterator_ops.py:358: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/contrib/rnn/python/ops/lstm_ops.py:696: to_int64 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.cast instead.
I Initializing variables...
I STARTING Optimization
I Training epoch 0...
I Finished training epoch 0 - loss: 389.854401
I Training epoch 1...
[...]
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/training/saver.py:966: remove_checkpoint (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.
Instructions for updating:
Use standard file APIs to delete files with this prefix.
I Training epoch 199...
I Finished training epoch 199 - loss: 7.553509
I FINISHED optimization in 0:00:42.070729
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/training/saver.py:1266: checkpoint_exists (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.
Instructions for updating:
Use standard file APIs to check for files with this prefix.
I Restored variables from most recent checkpoint at /root/.local/share/deepspeech/ldc93s1/train-200, step 200
Testing model on data/ldc93s1/ldc93s1.csv
I Test epoch...
Test on data/ldc93s1/ldc93s1.csv - WER: 0.000000, CER: 0.000000, loss: 3.357677

---

WER: 0.000000, CER: 0.000000, loss: 3.357677

- src: "she had your dark suit in greasy wash water all year"
- res: "she had your dark suit in greasy wash water all year"

---

root@javr:/srv/DeepSpeech#
```

## Error

```bash
tgingras@javr:/srv/DeepSpeech\$ ./DeepSpeech.py
/usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/dtypes.py:526: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
\_np_qint8 = np.dtype([("qint8", np.int8, 1)])
/usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/dtypes.py:527: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
\_np_quint8 = np.dtype([("quint8", np.uint8, 1)])
/usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/dtypes.py:528: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
\_np_qint16 = np.dtype([("qint16", np.int16, 1)])
/usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/dtypes.py:529: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
\_np_quint16 = np.dtype([("quint16", np.uint16, 1)])
/usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/dtypes.py:530: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
\_np_qint32 = np.dtype([("qint32", np.int32, 1)])
/usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/dtypes.py:535: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
np_resource = np.dtype([("resource", np.ubyte, 1)])
Traceback (most recent call last):
File "./DeepSpeech.py", line 19, in <module>
from evaluate import evaluate
File "/srv/DeepSpeech/evaluate.py", line 19, in <module>
from util.feeding import create_dataset
File "/srv/DeepSpeech/util/feeding.py", line 13, in <module>
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
File "/usr/local/lib/python3.6/dist-packages/tensorflow/contrib/**init**.py", line 30, in <module>
from tensorflow.contrib import cloud
File "/usr/local/lib/python3.6/dist-packages/tensorflow/contrib/cloud/**init**.py", line 28, in <module>
from tensorflow.contrib.bigtable.python.ops.bigtable_api import BigtableClient
File "/usr/local/lib/python3.6/dist-packages/tensorflow/contrib/bigtable/**init**.py", line 29, in <module>
from tensorflow.contrib.bigtable.python.ops.bigtable_api import BigtableClient
File "/usr/local/lib/python3.6/dist-packages/tensorflow/contrib/bigtable/python/ops/bigtable_api.py", line 44, in <module>
resource_loader.get_path_to_datafile("\_bigtable.so"))
File "/usr/local/lib/python3.6/dist-packages/tensorflow/contrib/util/loader.py", line 56, in load_op_library
ret = load_library.load_op_library(path)
File "/usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/load_library.py", line 61, in load_op_library
lib_handle = py_tf.TF_LoadLibrary(library_filename)
tensorflow.python.framework.errors_impl.NotFoundError: /usr/local/lib/python3.6/dist-packages/tensorflow/contrib/bigtable/python/ops/\_bigtable.so: undefined symbol: \_ZTIN10tensorflow4data15DatasetOpKernelE
```
