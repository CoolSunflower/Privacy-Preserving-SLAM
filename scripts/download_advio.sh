# Download all 23 data ZIPs from Zenodo
for i in $(seq -f "%02g" 13 16);
do
  wget -O advio-$i.zip https://zenodo.org/record/1476931/files/advio-$i.zip
done