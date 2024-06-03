#!/usr/bin/env bash
set -e

echo 'Change Zsh theme'
sed -i -e 's/ZSH_THEME="devcontainers"/ZSH_THEME="steeef"/g' ${HOME}/.zshrc

echo 'Disable Oh My Zsh plugin: Git'
sed -i -e 's/plugins=(git)/plugins=()/g' ${HOME}/.zshrc

echo 'Add env variables to zshrc'
echo '' >> ${HOME}/.zshrc
echo 'JAVA_HOME="/usr/local/sdkman/candidates/java/current"' >> ${HOME}/.zshrc
echo 'SPARK_HOME="/usr/local/sdkman/candidates/spark/current"' >> ${HOME}/.zshrc
echo 'export PATH="${JAVA_HOME}/bin:${PATH}"' >> ${HOME}/.zshrc
echo 'export PATH="${SPARK_HOME}/bin:${PATH}"' >> ${HOME}/.zshrc

echo 'Add aliases to zshrc'
echo '' >> ${HOME}/.zshrc
echo 'alias cp="cp -i"' >> ${HOME}/.zshrc
echo 'alias l="ls -CF --group-directories-first"' >> ${HOME}/.zshrc
echo 'alias ll="ls -lh --group-directories-first"' >> ${HOME}/.zshrc
echo 'alias la="ls -AF --group-directories-first"' >> ${HOME}/.zshrc
echo 'alias lla="ls -Alh --group-directories-first"' >> ${HOME}/.zshrc
echo 'alias gti="git"' >> ${HOME}/.zshrc
echo 'alias mm="make"' >> ${HOME}/.zshrc
echo 'alias mkae="make"' >> ${HOME}/.zshrc

echo 'Add bash command to zshrc: mkcdir'
echo '' >> ${HOME}/.zshrc
echo 'mkcdir() {' >> ${HOME}/.zshrc
echo '  mkdir -p -- "$1" && cd -P -- "$1"' >> ${HOME}/.zshrc
echo '}' >> ${HOME}/.zshrc

echo 'Add bash command to zshrc: gl'
echo '' >> ${HOME}/.zshrc
echo '# Show the last 12 git commits' >> ${HOME}/.zshrc
echo 'gl() {' >> ${HOME}/.zshrc
echo '  m=$(git l | wc --lines)' >> ${HOME}/.zshrc
echo '  number_of_commits=$((m + 1))' >> ${HOME}/.zshrc
echo '  if [ $number_of_commits -lt 12 ]' >> ${HOME}/.zshrc
echo '  then' >> ${HOME}/.zshrc
echo '    git l | sed "$number_of_commits"q' >> ${HOME}/.zshrc
echo '  else' >> ${HOME}/.zshrc
echo '    # quit after first 12 lines' >> ${HOME}/.zshrc
echo '    git l | sed 12q' >> ${HOME}/.zshrc
echo '  fi' >> ${HOME}/.zshrc
echo '}' >> ${HOME}/.zshrc

echo 'install pre-commit'
git config --global --add safe.directory /workspaces/onekit
cd /workspaces/onekit && sudo pre-commit install

echo 'Post creation script complete'
