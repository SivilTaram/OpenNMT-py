set MODEL=dial_bert_12_layer_pretrained_0.1_step_9000.pt
cd ..
onmt_translate -gpu 0 ^
               -batch_size 20 ^
               -beam_size 10 ^
               -model tmp/%MODEL% ^
               -src data/dial-src-val.txt ^
               -output data/%MODEL%.predict.txt ^
               -verbose ^
               -stepwise_penalty ^
               -coverage_penalty summary ^
               -beta 5 ^
               -length_penalty wu ^
               -alpha 0.9 ^
               -block_ngram_repeat 3 ^
               -ignore_when_blocking "." "</t>" "<t>"
