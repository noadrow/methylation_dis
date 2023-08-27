library(regioneR)
library("BSgenome.Hsapiens.UCSC.hg19")
library(qqman)

set.seed(123456)

createDataset <- function(num.snps=20000, max.peaks=5) {
  hg19.genome <- filterChromosomes(getGenome("hg19"))
  snps <- sort(createRandomRegions(nregions=num.snps, length.mean=1, 
                                   length.sd=0, genome=filterChromosomes(getGenome("hg19"))))
  names(snps) <- paste0("rs", seq_len(num.snps))
  snps$pval <- rnorm(n = num.snps, mean = 0.5, sd = 1)
  snps$pval[snps$pval<0] <- -1*snps$pval[snps$pval<0]
  #define the "significant peaks"
  peaks <- createRandomRegions(runif(1, 1, max.peaks), 8e6, 4e6)
  peaks
  for(npeak in seq_along(peaks)) {
    snps.in.peak <- which(overlapsAny(snps, peaks[npeak]))
    snps$pval[snps.in.peak] <- runif(n = length(snps.in.peak), 
                                     min=0.1, max=runif(1,6,8))
  }
  snps$pval <- 10^(-1*snps$pval)
  return(list(peaks=peaks, snps=snps))
}


ds <- createDataset()
ds$snps
manhattan(gwasResults, chr="CHR", bp="BP", snp="SNP", p="P" )