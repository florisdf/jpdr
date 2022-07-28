FROM nvcr.io/nvidia/cuda:11.4.2-devel-ubuntu20.04

RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install tmux zsh ffmpeg libsm6 libxext6 neovim python3-distutils python3-apt libpython3.8-dev sudo curl git -y
RUN curl https://bootstrap.pypa.io/get-pip.py | python3
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended
RUN chsh -s /usr/bin/zsh

RUN pip install ranger-fm pynvim
RUN update-alternatives --install /usr/bin/vim vim /usr/bin/nvim 60

RUN groupadd -g 2008 fdf
RUN useradd -ms /usr/bin/zsh -d /data/research/fdf -g fdf -G sudo -u 2008 fdf
USER fdf
WORKDIR /data/research/fdf

CMD /usr/bin/zsh
