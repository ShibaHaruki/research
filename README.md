cd OneDrive - 学校法人立命館\ドキュメント\研究コード
(Get-Content requirements.txt) -notmatch 'pywin32' | Set-Content requirements.txt
pip freeze > requirements.txt
git add .
git commit -m "Add requirements"
git pull --rebase origin main
git push origin main

ssh coder.sotuken.main

cd ~
git clone git@github.com:ShibaHaruki/research.git

cd ~/research
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pipreqs
pipreqs . --encoding=latin-1 --force
cd shiba_LSM
ls
pip list
