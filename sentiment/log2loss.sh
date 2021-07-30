cat  train.log |sed -n '/loss/p'|awk '{print $2}'|awk -F'(' '{print $2}'|awk -F',' '{print $1}' >lossList.txt
