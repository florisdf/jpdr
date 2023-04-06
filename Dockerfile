FROM determinedai/environments:cuda-11.1-base-gpu-0.19.1

RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install tmux zsh neovim python3-distutils python3-apt libpython3.8-dev sudo curl git libgl1 -y
RUN curl https://bootstrap.pypa.io/get-pip.py | python3
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended
RUN chsh -s /usr/bin/zsh

RUN pip install ranger-fm pynvim
RUN update-alternatives --install /usr/bin/vim vim /usr/bin/nvim 60
