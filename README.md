cd OneDrive - 学校法人立命館\ドキュメント\研究コード
git add .
git commit -m ""
git push

cd ~
git clone git@github.com:ShibaHaruki/research.git

cd ~/research
python3 -m venv .venv
source .venv/bin/activate
pip install brian2 numpy scipy matplotlib jupyter ipykernel
cd shiba_LSM

