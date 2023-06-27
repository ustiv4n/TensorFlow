for file in *.txt; do
mv "$file" "$(echo "$file" | sed 's/res//')"
done