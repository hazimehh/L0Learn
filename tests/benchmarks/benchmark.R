library("L0Learn")
library("rbenchmark")

L0LEARNVERSIONDATAFOLDER = normalizePath(file.path("~/Documents/GitHub/L0LearnVersionData"))
version_to_load_from="1.2.0"
time_version_to_load_from = paste('time', version_to_load_from, sep='_')

current_version = packageVersion("L0Learn")

# Assert data is available to test from
if (!(time_version_to_load_from %in% dir(L0LEARNVERSIONDATAFOLDER))){
    print(L0LEARNVERSIONDATAFOLDER)
    stop("'version_to_load_from' must exist in 'L0LEARNVERSIONDATAFOLDER'")
}


# Load data object
data_large <- readRDS(file.path(L0LEARNVERSIONDATAFOLDER, time_version_to_load_from, "data_large.rData"))
data_medium <- readRDS(file.path(L0LEARNVERSIONDATAFOLDER, time_version_to_load_from, "data_medium.rData"))
data_small <- readRDS(file.path(L0LEARNVERSIONDATAFOLDER, time_version_to_load_from, "data_small.rData"))

L0_grid <- readRDS(file.path(L0LEARNVERSIONDATAFOLDER, time_version_to_load_from, "L0_grid.rData"))
L012_grid <- readRDS(file.path(L0LEARNVERSIONDATAFOLDER, time_version_to_load_from, "L012_grid.rData"))

L0_nGamma = 1
L012_nGamma <- length(L012_grid)

# Load tests:
tests <- readLines(file.path(L0LEARNVERSIONDATAFOLDER, 
                             time_version_to_load_from, 'tests.txt'))

# Run tests:

for (i in 1:length(tests)){
    
    time <- eval(parse(text=tests[[i]]))
    version_time <- readRDS(file.path(L0LEARNVERSIONDATAFOLDER,
                                      time_version_to_load_from,
                                      paste(i, ".rData", sep='')))
    print(paste(tests[i], time$elapsed, version_time$result$elapsed, sep = "|"))
    
}
