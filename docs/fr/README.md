# Introduction

# Objectifs

Déployer **_DeepSpeech_** et **_Tensorflow_** sur un serveur **_Ubuntu 18.04_** qui ne supporte _pas AVX et SSE_. De plus, le training se fait uniquement avec le _CPU_.

# Matériel requis

- Un serveur avec Ubuntu 18.04 (VM ou Physique)
- Minimum 12GB de mémoire vive (RAM)
- Minimum 8 Coeurs (_Les tests ont été effectués avec ce CPU : Intel(R) Xeon(R) CPU L5520 @ 2.27GHz_)

> Le serveur physique utilisé est : "HP ProLiant DL160 G6" avec oVirt comme hyperviseur.  
> -- _La VM utilisée a 16GB de RAM et 8 Coeurs le disque est un SSD avec 30GB_

# Les scripts

Disponible sur Github : <a href="https://github.com/studiowebux/deepspeech" target="_blank" rel="noopener noreferrer">Github Deepspeech</a>

# Installation

# Objectif

Préparer l'environnement pour installer DeepSpeech et Tensorflow sur Ubuntu 18.04

# Étape 1 - Définir la version à déployer

| Version Mozilla/Tensorflow | Version Mozilla/DeepSpeech | Version Bazel | Testé | YAML         |
| -------------------------- | -------------------------- | ------------- | ----- | ------------ |
| r1.13                      | tags/v0.5.1                | 0.19.1        | OUI   | Main.yaml    |
| r1.15                      | tags/v0.7.0-alpha.3        | 0.24.1        | OUI   | Main_v2.yaml |

> Notez que seulement ces versions ont été testées.

Selon la version choisie, vous devez vous assurer d'avoir toutes les dépendances.

> Elles sont disponibles directement dans le github

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

Pour connaitre quelles versions sont compatibles avec votre version de tensorflow,

Par exemple, pour _Mozilla/Tensorflow R1.13_

```bash
cat /srv/tensorflow/configure.py | grep check_bazel_version
def check_bazel_version(min_version, max_version):
check_bazel_version('0.19.0', '0.21.0')
```

En bas de cette page, vous pouvez aussi trouver l'information : <a href="https://www.tensorflow.org/install/source" target="_blank" rel="noopener noreferrer">Tensorflow Build From Source</a>

# Étape 2 - Installer les outils sur la machine de déploiement (Contrôleur Ansible)

## Ansible

Le logiciel Ansible est utilisé pour déployer le tout,

Voici le lien pour la documentation officielle pour l'installer : <a href="https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html#installing-ansible-on-macos" target="_blank" rel="noopener noreferrer">Installation Ansible</a>

Pour valider que l'installation s'est effectuée correctement, lancer cette commande:

```bash
ansible --version
ansible-playbook --version
```

## Git

Installer git sur votre machine,
voici le lien pour l'installer : <a href="https://git-scm.com" target="_blank" rel="noopener noreferrer">Installation Git</a>

Pour valider que l'installation s'est effectuée correctement, lancer cette commande:

```bash
git --version
```

## Clé SSH

Il est fortement recommandé d'utiliser une clé SSH pour se connecter au serveur,

Pour générer une clé privée, lancer cette commande et suivez les instructions de la console,

```bash
ssh-keygen
```

Pour copier votre clé publique sur le serveur, lancer cette commande:

```bash
ssh-copy-id user@1.2.3.4
```

> Remplacer le 'user' et '1.2.3.4' pour vos informations

De cette manière, Ansible va pouvoir se connecter automatiquement au serveur en utilisant votre clé privée.

# Résumé

1. Votre poste de déploiement a tous les outils.
2. Vous avez toutes les dépendances pour lancer le script de déploiement selon la version désirée.

# Déploiement

# Objectif

Lancer le script "Main.yaml" pour déployer DeepSpeech et Tensorflow sur votre système.

# Commandes

Vous allez être invité à entrer le mot de passe SUDO (`--ask-become`)

## DeepSpeech 0.5.1 et Tensorflow 1.13 :

```bash
ansible-playbook -i 1.2.3.4, Scripts/Main.yaml --extra-vars='{"tensorflow_version": "r1.13","DeepSpeech_version": "tags/v0.5.1","bazel_version": "0.19.1"}'--ask-become
```

## DeepSpeech 0.7.0-alpha.3 et Tensorflow 1.15 :

```bash
ansible-playbook -i 1.2.3.4, Scripts/Main.yaml --ask-become
--extra-vars='{"tensorflow_version": "r1.15","DeepSpeech_version": "tags/v0.7.0-alpha.3","bazel_version": "0.24.1"}'
```

# Les scripts

Il y a au total 10 scripts (incluant 1 optionnel)

1. Le premier script (`Install.yaml`) installe toutes les dépendances pour effectuer le déploiement.

2. les 4 prochains scripts installent les dépendances pour utiliser DeepSpeech et compiler Tensorflow.

   1. `SOX.yaml`
   2. `KENLM.yaml`
   3. `SWIG.yaml`
   4. `BAZEL.yaml`

3. Le sixième script (`DEEPSPEECH.yaml`) clone le repository de _Mozilla/DeepSpeech_, change la branche pour la version désirée et télécharge les fichiers avec git-lfs (_environ 1.7GB pour la version 0.5.1 & 700MB pour la version 0.7.0-alpha.3_)

4. Le septième script (`TENSORFLOW.yaml`) clone le repository de _Mozilla/tensorflow_, change la branche pour la version désirée, puis lance les commandes pour compiler le tout. Celui-ci utilise un autre script (`answer_tensorflow.yaml`) pour répondre automatiquement aux questions posées lors de la configuration.

5. le huitième script (`TS_DEEPSPEECH.yaml`) compile et installe les fichiers dans le répertoire `DeepSpeech/native_client/`,

   - _Javascript_
   - _ctcdecode_
   - _deepspeech_
   - _Python_

6. Le dernier script (`Test.yaml`), ce script lance simplement le démo qui est fourni avec DeepSpeech. Celui-ci doit se terminer sans erreur.

Vous pouvez facilement lancer toutes les commandes manuellement:

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

Pour installer la version de tensorflow disponible en ligne,

> Il est préférable d'utiliser la version compilée pour votre système.

```bash
ansible-playbook -i 1.2.3.4, Scripts/FIX_TENSORFLOW.yaml --extra-vars '{"tensorflow_version":"r1.13"}' --ask-become
```

# Notes

- L'étape pour **_Compiler Tensorflow_** prend beaucoup de temps (si possible, ajoutez plus de CPU pour accélérer le processus, _par exemple, avec 12 Coeurs/Threads de 2.2 GHz ça prend environ 1h15_)
- Pour voir la progression ou connaitre l'état du système, vous pouvez ouvrir une session SSH et lancer la commande `top` ou `ps -aec` pour voir se qui roule sur le système. Il est aussi possible d'afficher la sortie du processus utilisé, `cat /proc/[process ID du de celui nommé bazel-real]/fd/2`

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

# Problèmes survenus

# Incident #1 - Tensorflow compilé avec AVX

> Le système ne supporte pas cette fonctionnalité (AVX ni SSE)

## Solution #1

Utiliser la version compilée par le script `TENSORFLOW.yaml`

Pour connaitre les versions de bazel supportées,

```bash
cat /srv/tensorflow/configure.py | grep check_bazel_version
```

## Solution #2

Utiliser si possible une version disponible en ligne correspondant à votre besoin, <a href="https://github.com/yaroslavvb/tensorflow-community-wheels/issues" target="_blank" rel="noopener noreferrer">Github</a>

Cette solution est celle appliquée dans le script `FIX_TENSORFLOW.yaml`

<a href="https://github.com/Tzeny/tensorflowbuilds/raw/master/tensorflow-1.13.1-cp36-cp36m-linux_x86_64.whl" target="_blank" rel="noopener noreferrer">tensorflow-1.13.1-cp36-cp36m-linux_x86_64.whl</a> -
<a href="https://github.com/yaroslavvb/tensorflow-community-wheels/issues/111" target="_blank" rel="noopener noreferrer">Source</a>

Pour installer le package,

```bash
wget https://github.com/Tzeny/tensorflowbuilds/raw/master/tensorflow-1.13.1-cp36-cp36m-linux_x86_64.whl -O /tmp/tensorflow-1.13.1-cp36-cp36m-linux_x86_64.whl
pip3 uninstall tensorflow
pip3 install /tmp/tensorflow-1.13.1-cp36-cp36m-linux_x86_64.whl
```

# Erreur

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

# Incident #2 - Numpy retourne une erreur

La version de numpy qui est installé n'est pas assez récente. il suffit de mettre à jour numpy à la version la plus récente,

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

## Erreur

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

# Incident #3 - Aucun `ds_ctcdecoder`

Ce module doit être compilé, cette tâche est effectuée dans le script `TS_DEEPSPEECH.yaml`

> Cette erreur est survenue lors des tests manuels, le script n'avait pas été exécuté au moment de l'erreur.

Pour installer le module manuellement,

```bash
cd /srv/DeepSpeech/native_client/ctcdecode
make bindings NUM_PROCESSES=8
pip3 install dist/\*.whl
```

# Erreur

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

# Incident #4 - kenlm retourne une erreur

Cette erreur est causée par `git-lfs`. Lorsque le repository de _Mozilla/DeepSpeech_ a été cloné, les fichiers LFS n'ont pas été téléchargés.

> Cette erreur est survenue lors des tests manuels.

## Solution #1

Lancer les scripts correctement, l'étape pour télécharger les fichiers LFS se trouve dans le script : `DEEPSPEECH.yaml`

## Solution #2

Lancer les commandes suivantes:

> Changer la version du checkout pour votre version

```bash
git clone https://github.com/mozilla/DeepSpeech.git
cd /srv/DeepSpeech
git checkout tags/v0.5.1
git-lfs pull
```

Suite à ce téléchargement, le test est fonctionnel.

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

# Erreur

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

# Incident #5 - version tensorflow compilée localement

> Essai #1 et #5 sont PASS

## Essai #1 (PASS)

Cette version fonctionne : <a href="https://github.com/yaroslavvb/tensorflow-community-wheels/issues/111" target="_blank" rel="noopener noreferrer">Github</a>

> Mais l'objectif est de build tensorflow depuis les sources

## Essai #2 (FAIL)

La version de _bazel_ : **0.21.0**  
La version de _tensorflow_ : **Mozilla/tensoflow** _checkout_ : **r1.13**

la commande utilisée pour compiler:

```bash
bazel build --config=monolithic -c opt --copt=-O3 --copt="-D_GLIBCXX_USE_CXX11_ABI=0" --copt=-fvisibility=hidden //tensorflow/tools/pip_package:build_pip_package //native_client:libdeepspeech.so //native_client:generate_trie
```

## Essai #3 (FAIL)

Installer la version disponible,

```bash
pip3 install tensorflow==1.13.1
```

### Le résultat

```bash
tgingras@javr:/srv/DeepSpeech\$ ./DeepSpeech.py
2020-04-20 01:19:24.380911: F tensorflow/core/platform/cpu_feature_guard.cc:37] The TensorFlow library was compiled to use AVX instructions, but these aren't available on your machine.
Aborted (core dumped)
```

## Essai #4 (FAIL)

Mettre la version de bazel à 0.25.2
Et lancer la commande fournie sur le site de tensorflow

> L'erreur retournée : la version de bazel doit être entre 0.19.0 et 0.21.0 pour compiler la version de Mozilla/Tensorflow r1.13

```bash
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
```

## Essai #5 (PASS)

Avec la version de bazel 0.19.1, le build et le package ont été installés avec succès

Lien pour bazel 0.19.1 : <a href="https://github.com/bazelbuild/bazel/releases/download/0.19.1/bazel_0.19.1-linux-x86_64.deb" target="_blank" rel="noopener noreferrer">Github</a>

Voici les commandes exécutées:

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

> La section `apt` a été nécessaire, car un conflit de version du package est survenu.

### Bundle

#### Version de tensorflow

```bash
root@javr:/srv/tensorflow# git status
On branch r1.13
Your branch is up to date with 'origin/r1.13'.
```

#### Version de bazel

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

#### Lancement du démo de DeepSpeech

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

## Erreur

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

# Incident #6 - Le package pip de tensorflow ne build pas

Ce playbook `TS_DEEPSPEECH.yaml`,
peut retourner une erreur lors de la tache : ```Build Tensorflow \`build_pip_package\````

> J'ai ajouté un `ignore_errors` pour continuer le déploiement,
> Si le test est PASS, alors vous pouvez ignorer cette erreur.

La solution est de lancer les commandes manuellement dans la console SSH,

```bash
cd /srv/tensorflow
./configure # I answered NO to all questions and use the default options for Python

bazel build --config=opt --config=noaws --config=nogcp --config=nohdfs --config=noignite --config=nokafka --config=nonccl //tensorflow/tools/pip_package:build_pip_package

./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow

pip3 uninstall -y tensorflow
pip3 install /tmp/tensorflow/*.whl
```

> Si vous connaissez la raison de cette erreur, faites-moi savoir, merci.
