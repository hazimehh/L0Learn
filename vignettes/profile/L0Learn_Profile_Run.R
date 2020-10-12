# CMD_BASE <- '/bin/bash -c "mprof run Rscript L0Learn_Profile.R --n %s --p %s --k %s --s %s --t %s --w %s --m %s --f %s\"'
MD_BASE <- "mprof run Rscript L0Learn_Profile.R --n %s --p %s --k %s --s %s --t %s --w %s --m %s --f %s"
print(CMD_BASE)
run <- list(n=1000, p=10000, k=10, s=1, t=2.1, w=4, m=1, f="test_run")

cmd <- sprintf(CMD_BASE, run$n, run$p, run$k, run$s, run$t, run$w, run$m, run$f)
print(cmd)
Sys.setenv(SHELL = "/bin/bash")
system(cmd) # Creates <run$f>.dat and <run$f>.csv files in same directory as file.

memory_usage <- read.table(paste(run$f, ".dat", sep=''), header=TRUE, skip=3) # load in data file
timing <- read.table(paste(run$f, ".csv", sep=''))


old_path <- Sys.getenv("PATH")
Sys.setenv(PATH = paste("$HOME/anaconda/bin", old_path, sep = ":"))