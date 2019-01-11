# python main.py --status train \
# 		--train data/Onto/train.char.bmes \
#		--dev data/Onto/dev.char.bmes \
#		--test data/Onto/test.char.bmes \
#		--savemodel data/Onto/test_shared_attn/saved_model \

python main.py --status train \
		--train data/MSRA/train.txt \
		--dev data/MSRA/test.txt \
		--test data/MSRA/test.txt \
		--savemodel data/MSRA/attn_shared_mul/saved_model \


# python main.py --status decode \
# 		--raw ../data/onto4ner.cn/test.char.bmes \
# 		--savedset ../data/onto4ner.cn/saved_model \
# 		--loadmodel ../data/onto4ner.cn/saved_model.13.model \
# 		--output ../data/onto4ner.cn/raw.out \
