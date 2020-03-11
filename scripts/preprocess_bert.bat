cd ..
onmt_preprocess -train_src data/dial-src-train-bert.txt ^
-train_tgt data/dial-tgt-train-bert.txt ^
-valid_src data/dial-src-val-bert.txt ^
-valid_tgt data/dial-tgt-val-bert.txt ^
-save_data data/dial_bert ^
-src_vocab bert-base-chinese/vocab.txt ^
-tgt_vocab bert-base-chinese/vocab.txt ^
-dynamic_dict ^
-share_vocab ^
-use_bert_tokenize