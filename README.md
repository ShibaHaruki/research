cd "C:\Users\elast\OneDrive - 学校法人立命館\ドキュメント\研究コード"
git add requirements.txt
git add .
git commit -m "Add requirements"
git pull --rebase origin main
git push origin main

ssh coder.Haru.main

cd ~
git clone git@github.com:ShibaHaruki/research.git


cd ~/research
git pull origin main
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pipreqs
pipreqs . --encoding=latin-1 --force
pip list
ls
cd shiba_LSM

exit
