i=1
for filename in *.txt; do
    newname="$(printf "%s\n" "$i" | awk '{printf "%d", $0 + 90}')".txt
    mv "$filename" "$newname"
    i=$((i+1))
done