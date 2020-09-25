cd ./results/fc
du -a | cut -d/ -f2 | sort | uniq -c | sort -nr
