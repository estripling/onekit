#!/usr/bin/env bash
set -e

echo 'Disable Oh My Zsh plugin: Git'
sed -i -e 's/plugins=(git)/plugins=()/g' ${HOME}/.zshrc

echo 'Add env variables to zshrc'
echo 'JAVA_HOME="/usr/local/sdkman/candidates/java/current"' >> ${HOME}/.zshrc
echo 'SPARK_HOME="/usr/local/sdkman/candidates/spark/current"' >> ${HOME}/.zshrc
echo 'export PATH="${JAVA_HOME}/bin:${PATH}"' >> ${HOME}/.zshrc
echo 'export PATH="${SPARK_HOME}/bin:${PATH}"' >> ${HOME}/.zshrc

echo 'Add aliases to zshrc'
echo 'alias cp="cp -i"' >> ${HOME}/.zshrc
echo 'alias l="ls -CF --group-directories-first"' >> ${HOME}/.zshrc
echo 'alias ll="ls -lh --group-directories-first"' >> ${HOME}/.zshrc
echo 'alias la="ls -AF --group-directories-first"' >> ${HOME}/.zshrc
echo 'alias lla="ls -Alh --group-directories-first"' >> ${HOME}/.zshrc
echo 'alias gti="git"' >> ${HOME}/.zshrc
echo 'alias mm="make"' >> ${HOME}/.zshrc
echo 'alias mkae="make"' >> ${HOME}/.zshrc

echo 'Post creation script complete'
