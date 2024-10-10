
git filter-branch --force --index-filter \
"git rm --cached --ignore-unmatch Data/ComboDrugGrowth_Nov2017.csv" \
--prune-empty --tag-name-filter cat -- --all

