library("testthat")
library("L0Learn")

L0LEARNVERSIONDATAFOLDER = normalizePath(file.path("~/Documents/GitHub/L0LearnVersionData"))
version_to_load_from="1.2.0"
current_version = packageVersion("L0Learn")

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
        # NOTE: Between v1.2.0 and v1.2.1 a change in standardization results
        # in numerical precision issues on the order of ~1e-5
        
        if ((version_to_load_from == '1.2.0') && (current_version != '1.2.0')){
            tolerance = 1e-5
        } else {
            tolerance = 1e-9
        }
        fit <- eval(parse(text=tests[[i]]))
        version_fit <- readRDS(file.path(L0LEARNVERSIONDATAFOLDER,
                                         version_to_load_from,
                                         paste(i, ".rData", sep='')))
        expect_equal(fit, version_fit$fit, info=i, tolerance=tolerance)
    }
})