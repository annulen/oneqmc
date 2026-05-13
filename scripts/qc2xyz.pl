#!/usr/bin/env perl

use JSON::XS;

use strict;
use warnings;

my $json_data = do { open my $fh, '<', $ARGV[0]; local $/; <$fh> };
#open my $fh, '<', $ARGV[0];
my $json = decode_json $json_data;
for (my $i = 0; my $mol = $json->{$i}; $i++) {
    my @symbols = $mol->{symbols}->@*;
    my @geometry = $mol->{geometry}->@*;
    3 * scalar @symbols == scalar @geometry or die "Mismatch between symbols and geometry";
    #print "$i  @symbols  @geometry\n";
    print scalar @symbols, "\n";
    print "Molecule $i\n";
    for (my $j = 0; $j < @symbols; $j++) {
        printf "%s %f %f %f\n", $symbols[$j], @geometry[3 * $j .. 3 * $j + 2];
    }
}
