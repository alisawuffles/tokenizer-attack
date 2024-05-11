# make sure to go into the directory where you want the data to be downloaded

wget 'https://data.together.xyz/redpajama-data-1T/v1.0.0/urls.txt'
while read line; do
    if [[ $line == *"github"* ]]; then
        dload_loc=${line#https://data.together.xyz/redpajama-data-1T/v1.0.0/}
        mkdir -p $(dirname $dload_loc)
        wget "$line" -O "$dload_loc"
    fi
done < urls.txt

wget 'https://data.together.xyz/redpajama-data-1T/v1.0.0/urls.txt'
while read line; do
    if [[ $line == *"wikipedia"* ]]; then
        dload_loc=${line#https://data.together.xyz/redpajama-data-1T/v1.0.0/}
        mkdir -p $(dirname $dload_loc)
        wget "$line" -O "$dload_loc"
    fi
done < urls.txt

wget 'https://data.together.xyz/redpajama-data-1T/v1.0.0/urls.txt'
while read line; do
    if [[ $line == *"common_crawl"* ]]; then
        dload_loc=${line#https://data.together.xyz/redpajama-data-1T/v1.0.0/}
        mkdir -p $(dirname $dload_loc)
        wget "$line" -O "$dload_loc"
    fi
done < urls.txt