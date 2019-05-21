# Build Docker image

```
docker build -t bertsum .
```

# Download the author processed data
```
cd bert_data
mkdir downloaded
cd downloaded
curl -c /tmp/cookies "https://drive.google.com/uc?id=1-NJKRoNzk2ugjjB2mfnmpR15KYB_Fr0s&export=download" > /tmp/intermezzo.html
curl -L -b /tmp/cookies "https://drive.google.com$(cat /tmp/intermezzo.html | grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')" > bertsum_data.zip
unzip bertsum_data.zip
rm bertsum_data.zip
cd ../..
```

# Regenerate data

## Download tools
```
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip
unzip -o -q stanford-corenlp-full-2018-10-05.zip
rm stanford-corenlp-full-2018-10-05.zip
```

## Download raw data

```
cd raw_data
mkdir cnndm

# CNN
curl -c /tmp/cookies "https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfTHk4NFg2SndKcjQ" > /tmp/intermezzo.html
curl -L -b /tmp/cookies "https://drive.google.com$(cat /tmp/intermezzo.html | grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')" > cnn_stories.tgz
tar zxf cnn_stories.tgz -C cnndm --strip-components=2
rm cnn_stories.tgz

# Dailymail
curl -c /tmp/cookies "https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfM1BxdkxVaTY2bWs" > /tmp/intermezzo.html
curl -L -b /tmp/cookies "https://drive.google.com$(cat /tmp/intermezzo.html | grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')" > dailymail_stories.tgz
tar zxf dailymail_stories.tgz -C cnndm --strip-components=2
rm dailymail_stories.tgz

mv cnndm/stories stories
rm -r cnndm
cd ..
```

## Process data in Docker
```
docker run --rm -it -v $(pwd):/workspace bertsum
export CLASSPATH=$(pwd)/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar

python src/preprocess.py -mode fix_missing_period -raw_path raw_data/stories -save_path raw_data/period_fixed
python src/preprocess.py -mode tokenize -raw_path raw_data/period_fixed -save_path raw_data/tokens
python src/preprocess.py -mode format_to_lines -raw_path raw_data/tokens -save_path raw_data/json_data/cnndm -map_path urls -lower

# bert-base-uncased
python src/preprocess.py -mode format_to_bert -raw_path raw_data/json_data -save_path bert_data/bert_base_uncased -oracle_mode greedy -n_cpus 4 -bert_model bert-base-uncased

# bert-large-uncased
python src/preprocess.py -mode format_to_bert -raw_path raw_data/json_data -save_path bert_data/bert_large_uncased -oracle_mode greedy -n_cpus 4 -bert_model bert-large-uncased

# sentence scores instead of labels, bert-large-uncased
python src/preprocess.py -mode format_to_bert_w_scores -raw_path raw_data/json_data -save_path bert_data/bert_large_uncased_w_scores -n_cpus 4 -bert_model bert-large-uncased

# fairseq data format
python src/preprocess.py -mode format_to_fairseq -raw_path raw_data/tokens -save_path fairseq_data/trunc400 -map_path urls -n_cpus 4 -max_src_ntokens 400
python src/preprocess.py -mode format_to_fairseq -raw_path raw_data/tokens -save_path fairseq_data/no_trunc -map_path urls -n_cpus 4 -max_src_ntokens -1
```
