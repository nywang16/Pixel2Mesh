chmod -R 777 .git/
rm -r .git/
git init
git config user.email "nywang16@fudan.edu.cn"
git config user.name "nywang16"
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/nywang16/Pixel2Mesh.git
git push -f origin master
