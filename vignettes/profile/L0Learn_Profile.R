#!/usr/bin/env Rscript

library("optparse")

option_list <- list( 
    make_option("--n", action="store", default=1000, help="Number of rows [default]", type="integer"),
    make_option("--p", action="store", default=10000, help="Number of columns [default]", type="integer"),
    make_option("--k", action="store", default=10, help="Number of positive examples [default]", type="integer"),
    make_option("--s", action="store", default=1, help="seed [default]", type="integer"),
    make_option("--t", action="store", default=2.1, help="Threshold for 0 values [default]", type="double"),
    make_option("--w", action="store", default=10, help="Wait time in seconds [default]", type="double"),
    make_option("--m", action="store", default=2, help="Wait mutliplier after matrix creation [default]", type="double"),
    make_option("--f", action="store", default=NULL, help="File name to save timing data to", type="character")
    )

library("Matrix")
library("L0Learn")
                                        
# get command line options, if help option encountered print help and exit,
# otherwise if options not found on command line then set defaults, 
opt <- parse_args(OptionParser(option_list=option_list))

if (is.null(opt$f)) {
  stop("f [Filename] parameter must be provided. See script usage (--help)")
}

times <- data.frame(step= c('Start'),
                    time = c(as(Sys.time(), "double")),
                    stringsAsFactors = FALSE)

log <- function(times, event){
	time <- as(Sys.time(), "double")
	times = rbind(times, list(event, time))
	times
}


Sys.sleep(opt$w)
times <- log(times, "GenSynthetic Start")
tmp <- L0Learn::GenSynthetic(opt$n, opt$p, opt$k, opt$s, opt$t)
times <- log(times, "GenSynthetic End")

Sys.sleep(opt$w)

times <- log(times, "L0Learn Dense Start")
L0Learn.fit(tmp[[1]], tmp[[2]], intercept = FALSE)
times <- log(times, "L0Learn Dense End")

Sys.sleep(opt$w)

times <- log(times, "as dgCMatrix Start")
x_sparse <- as(tmp[[1]], "dgCMatrix")
y <- as.matrix(tmp[[2]])
rm(tmp)
times <- log(times, "as dgCMatrix End")

Sys.sleep(opt$w)

times <- log(times, "L0Learn Sparse Start")
L0Learn.fit(x_sparse, y, intercept = FALSE)
times <- log(times, "L0Learn Sparse End")
Sys.sleep(opt$w)

times <- log(times, "Stop")
write.csv(x=times, file=paste(opt$f, ".csv", sep=''))
