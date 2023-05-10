python3 train.py -d ../SemEval-task5/df_train.csv --trial ../SemEval-task5/df_test.csv -s ../sentiment-
datasets/train_E6oV3lV.csv --word_list ../word-list/word_all.txt --emb /kaggle/input/glove6b300dtxt/glove.6B.300d.txt -o output_dir -b 512 --epochs 1 --lr 0.002 --maxlen 50 -t H
HMM_transformer