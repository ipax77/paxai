use strict;
use warnings;


my @actions = (3375, 1724, 2695, 3125, 2089, 1686, 122, 3093, 177, 2698, 1201, 471, 2855, 6, 2088);
my $value = 0;
foreach my $action (@actions) {
    print $action . "\n";
    my $num = 0;
    my $unit = "";
    for (my $i = 0; $i < 3; $i++) {
        for (my $j = 0; $j < 20; $j++) {
            for (my $k = 0; $k < 60; $k++) {
                if ($num == $action) {
                    if ($i == 0) {
                        $unit = "Marine";
                        $value += 50;
                    } elsif ($i == 1) {
                        $unit = "Marauder";
                        $value += 95;
                    } elsif ($i == 2) {
                        $unit = "Reaper";
                        $value += 65;
                    }
                    $unit .= " $j,$k";
                    print $unit . "\n";
                }
                $num++;
            }
        }
    }
}
print $value . "\n";
