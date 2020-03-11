cd ..
onmt_preprocess -train_src data/dial-src-toy.txt ^
-train_tgt data/dial-tgt-toy.txt ^
-valid_src data/dial-src-toy.txt ^
-valid_tgt data/dial-tgt-toy.txt ^
-save_data data/dial_debug ^
-dynamic_dict ^
-share_vocab