library("Matrix")
library("testthat")
library("L0Learn")
source("utils.R")


tmp <-  L0Learn::GenSynthetic(n=100, p=1000, k=10, seed=1)
X <- tmp[[1]]
y <- tmp[[2]]
y_bin = sign(y)
tol = 1e-4

if (sum(apply(X, 2, sd) == 0)) {
    stop("X needs to have non-zero std for each column")
}

X_sparse <- as(X, "dgCMatrix")
seed = 1

VERSION_FOLDERS = "../L0Learn/version_tests"
version_to_load_from="1.1012"
#Assert in version folder

data <- load_data(version_to_load_from) # Loads in X, y
runs <- load_runs(version_to_load_from) # Loads in a list like
# runs[[1]] = list(eval="L0Learn.fit(...), fit=fit_object)
# run2

matrices = list(X=X, X_sparse=X_sparse)

if (packageVersion("L0Learn") == "2.0.0"){
    # savetest(seed, func, x, y, params, name)
    
    # Standard Tests
    # saved as "<method>_<matrix>_<penalty>.Rdata"
    for (f in c("fit", "cvfit")){
        for (m_name in names(matrices)){
            m = matrices[[m_name]]
            for (p in c("L0", "L0L2", "L0L1")){
                savetest(seed, f, m, y, list(penalty=p, intercept=FALSE),
                         paste(paste(f, m_name, p, sep="_"), ".Rdata", sep=''))
            }
        }
    }
    
    # Test Losses
    # saved as "fit_<matrix>_<loss>.Rdata"
    for (l in c("SquaredError")){
        for (m_name in names(matrices)){
            m = matrices[[m_name]]
            for (p in c("L0", "L0L2", "L0L1")){
                savetest(seed, 'fit', m, y, list(loss=l, intercept=FALSE),
                         paste(paste('fit', m_name, l, sep="_"), ".Rdata", sep=''))
            }
        }
    }
    
    for (l in c("Logistic", "SquaredHinge")){
        for (m_name in names(matrices)){
            m = matrices[[m_name]]
            for (p in c("L0", "L0L2", "L0L1")){
                savetest(seed, f, m, y_bin, list(loss=l, intercept=FALSE),
                         paste(paste(f, m_name, l, sep="_"), ".Rdata", sep=''))
            }
        }
    }
    
    # Algorithms
    # saved as "fit_<matrix>_<penalty>.Rdata"
    for (a in c("CD", "CDPSI")){
        for (m_name in names(matrices)){
            m = matrices[[m_name]]
            for (p in c("L0", "L0L2", "L0L1")){
                savetest(seed, "fit", m, y, list(algorithm=a, intercept=FALSE),
                         paste(paste("fit", m_name, a, sep="_"), ".Rdata", sep=''))
            }
        }
    }
    
} else if (packageVersion("L0Learn") == "2.0.0") {
    
    # Standard Tests
    for (f in c("fit", "cvfit")){
        for (m_name in names(matrices)){
            m = matrices[[m_name]]
            for (p in c("L0", "L0L2", "L0L1")){
                load(paste(paste(f, m_name, p, sep="_"), ".Rdata", sep=''))
                runtest(savedResult)
            }
        }
    }
    
    # Test Losses
    # saved as "fit_<matrix>_<loss>.Rdata"
    for (l in c("SquaredError")){
        for (m_name in names(matrices)){
            m = matrices[[m_name]]
            for (p in c("L0", "L0L2", "L0L1")){
                load(paste(paste('fit', m_name, l, sep="_"), ".Rdata", sep=''))
                runtest(savedResult)
            }
        }
    }
    
    for (l in c("Logistic", "SquaredHinge")){
        for (m_name in names(matrices)){
            m = matrices[[m_name]]
            for (p in c("L0", "L0L2", "L0L1")){
                load(paste(paste(f, m_name, l, sep="_"), ".Rdata", sep=''))
                runtest(savedResult)
            }
        }
    }
    
    # Algorithms
    # saved as "fit_<matrix>_<penalty>.Rdata"
    for (a in c("CD", "CDPSI")){
        for (m_name in names(matrices)){
            m = matrices[[m_name]]
            for (p in c("L0", "L0L2", "L0L1")){
                load(paste(paste("fit", m_name, a, sep="_"), ".Rdata", sep=''))
                runtest(savedResult)
            }
        }
    }
    
}