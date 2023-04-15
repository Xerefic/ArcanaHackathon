companies=`ls ../FMP`

process(){
    # rm -f FMP/$1/price.csv
    touch FMP/$1/price.csv
    echo "ds,symbol,close,volume" >> FMP/$1/price.csv
    for price in `ls prices`;
    do
        cat prices/$price | grep ",$1," | sort -M -k 1 >> FMP/$1/price.csv
    done
} 
export -f process

printf "%s\n" "${companies[@]}" | xargs -P 128 -I {} bash -c 'process "$@"' _ {}