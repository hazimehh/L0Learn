library("testthat")
library("L0Learn")

L0LEARNVERSIONDATAFOLDER = normalizePath(file.path("~/Documents/GitHub/L0LearnVersionData"))
version_to_load_from="1.2.1"

# Assert data is available to test from
if (!(version_to_load_from %in% dir(L0LEARNVERSIONDATAFOLDER))){
    print(L0LEARNVERSIONDATAFOLDER)
    stop("'version_to_load_from' must exist in 'L0LEARNVERSIONDATAFOLDER'")
}


test_that("All versions run as expected", {
    # Load data object
    data <- readRDS(file.path(L0LEARNVERSIONDATAFOLDER, version_to_load_from, "data.rData"))
    
    # Load tests:
    tests <- readLines(file.path(L0LEARNVERSIONDATAFOLDER, 
                       version_to_load_from, 'tests.txt'))
    
    
    
    # Run tests:
    for (i in 1:length(tests)){
        fit <- eval(parse(text=tests[[i]]))
        version_fit <- readRDS(file.path(L0LEARNVERSIONDATAFOLDER,
                                         version_to_load_from,
                                         paste(i, ".rData", sep='')))
        expect_equal(fit, version_fit$fit, info=i)
    }
})