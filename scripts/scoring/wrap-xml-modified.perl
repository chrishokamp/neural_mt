#!/usr/bin/perl -w

use strict;

if (scalar @ARGV <= 2){
    print STDOUT "Usage:\nperl wrap-xml-modified.perl language source_file.sgm Engine flag < in_txt_file > out_file.smg\n";
    print STDOUT "where flag = {src, ref, tst}\nExample:\n";
    print STDOUT "perl wrap-xml-modified.perl en newstest2013-src.es.sgm moses ref < newstest2013.en > newstest2013-ref.en.sgm\n\n";
    exit;
}
my ($language,$src,$system,$flag) = @ARGV;
die("wrapping frame not found ($src)") unless -e $src;
$system = "moses" unless $system;

open(SRC,$src);
my @OUT = <STDIN>;
chomp(@OUT);
my $sid = "SETID";
my $cnt = 0;
if ($flag eq "src"){
    while(my $line = <SRC>){
        if ($cnt eq 0){
            print "<srcset setid=\"" . $sid . "\" srclang=\"any\">\n<DOC sysid=\"Source\" docid=\"" .$src . "\" origlang=\"" . $language . "\">\n<p>\n";
        }
        ++$cnt;
        $line =~ s/\n$//;
        print "<seg id=\"" . $cnt . "\">" . $line . "</seg>\n";
    }
    print "</DOC>\n</" . $flag . "set>";
    exit;
}

while(<SRC>) {
    chomp;
    if (/^<srcset/) {
	s/<srcset/<${flag}set trglang="$language"/;
    }
    elsif (/^<\/srcset/) {
	s/<\/srcset/<\/${flag}set/;
    }
    elsif (/^<DOC/i) {
	s/<DOC/<DOC sysid="$system"/i;
    }
    elsif (/<seg/) {
        my $line = shift(@OUT);
        $line = "" if $line =~ /NO BEST TRANSLATION/;
        if (/<\/seg>/) {
	  s/(<seg[^>]+> *).*(<\/seg>)/$1$line$2/;
        }
        else {
	  s/(<seg[^>]+> *)[^<]*/$1$line/;
        }
    }
    print $_."\n";
}
