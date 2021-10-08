#!/bin/bash
clear
source ~/.bashrc
conda activate w
file_name=$(date  "+%Y%m%d%H%M%S")

echo python Model.py ${file_name} 2>&1 | tee -a log/${file_name}.log
python Model.py ${file_name} 2>&1 | tee -a log/${file_name}.log

echo zip -9 -r zips/${file_name}.zip *.py *.sh results/${file_name}* log/${file_name}.log model/${file_name}* | tee -a log/${file_name}.log
zip -9 -r zips/${file_name}.zip *.py *.sh results/${file_name}* log/${file_name}.log model/${file_name}* | tee -a log/${file_name}.log

# echo rm -r -v results/* | tee -a log/${file_name}.log
# rm -r -v results/* | tee -a log/${file_name}.log

echo -e "\n"
