# Notes (WIP)

### Convert a 44100 wav file to 16000

```bash
for i in {1..256}; do sox ../audios/$i.wav --encoding signed-integer --rate 16000 ./out$i.wav; done
for i in {1..256}; do mv out$i.wav $i.wav; done
```

### alphabet

```text
#Alphabet

a
b
c
d
e
f
g
h
i
j
k
l
m
n
o
p
q
r
s
t
u
v
w
x
y
z
'
-
é
è
à
â
ê
ù
ç
ô
û
Û
"
!
?
ï
Ï
:
,
.
;
@
0
1
2
3
4
5
6
7
8
9
A
B
C
D
E
F
G
H
I
J
K
L
M
N
O
P
Q
R
S
T
U
V
W
X
Y
Z
É
È
À
Â
Ê
Ù
Ç
Ô
# The last (non-comment) line needs to end with a newline.

```

### data.csv

```csv
wav_filename,wav_filesize,transcript
/opt/javr/audios/1.wav,267106,Est-ce que tu peux te connecter sur Facebook ?
/opt/javr/audios/2.wav,267106,Est-ce que tu peux te connecter sur twitter ?
```

### training.sh

```bash
#!/bin/sh
set -xe
if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

if [ -d "${COMPUTE_KEEP_DIR}" ]; then
    checkpoint_dir=$COMPUTE_KEEP_DIR
else
    checkpoint_dir="/opt/javr/checkpoints"
fi

# Force only one visible device because we have a single-sample dataset
# and when trying to run on multiple devices (like GPUs), this will break
export CUDA_VISIBLE_DEVICES=0

python -u DeepSpeech.py --noshow_progressbar \
  --train_files /opt/javr/data.csv \
  --test_files /opt/javr/test.csv \
  --train_batch_size 1 \
  --test_batch_size 1 \
  --n_hidden 100 \
  --epochs 200 \
  --export_dir /opt/javr/models/ \
  --alphabet_config_path /opt/javr/alphabet.0.txt \
  --checkpoint_dir "$checkpoint_dir" \
  "$@"
```

### lm.binary

Todo: rewrote the actual python script with our corpus ...

```bash
sudo nano vocabulary.txt
sudo lmplz --text vocabulary.txt --arpa words.arpa --o 5
cat words.arpa
sudo build_binary -T -s words.arpa lm.binary
```
