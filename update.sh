#!/usr/bash

echo '# note_md 使用markdown格式存放个人笔记' > README.md
echo '\n' >> README.md
echo '<pre>' >> README.md

cat>>README.md<<EOF
$(tree -DNtr ./)
EOF

echo '\n' >> README.md
echo '</pre>' >> README.md

