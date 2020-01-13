use strict;
use warnings;
use GD::Graph::lines;

my $file = "/data/ml/games_q.txt";
my $start = 1179;
my $step = 333;

my %data;
$data{0}{'WIN'} = 0;
$data{0}{'DRAW'} = 0;
$data{0}{'LOS'} = 0;

my $i = 0;
my $j = 0;
my $k = 0;
my $sum = 0;
open(FILE, "<", $file) or die $!;
while (<FILE>) {
    $i++;
    if ($i > $start) {
        if ($_ =~ /([^X]+)XXX/) {
        	if ($j >= $step) {
        		$j = 0;
        		$k++;     
        		$data{$k}{'WIN'} = 0;
		    	$data{$k}{'DRAW'} = 0;
		    	$data{$k}{'LOS'} = 0;
				   		
        	}
		    $j++;
		    if ($1 >=1) {
			    $data{$k}{'WIN'}++;
		    } elsif ($1 <= -0.3) {
		    	$data{$k}{'LOS'}++;
        	} else {
        		$data{$k}{'DRAW'}++;
        	}
		    
        }
    }
}

my @wins;
my @loss;
my @draws;

foreach my $ent (sort {$a <=> $b} keys %data) {
	push(@wins, int($data{$ent}{'WIN'} / $step * 100));
	push(@loss, int($data{$ent}{'LOS'} / $step * 100));
	push(@draws, int($data{$ent}{'DRAW'} / $step * 100));
	#print $ent . " => " . $data{$ent}{'WIN'} . "\n";
	
}

my @xaxis;
for (my $i = 0; $i <= $k; $i++) {
	push(@xaxis, $i);
}

my @pldata = (\@xaxis, \@wins, \@loss, \@draws);

my $mygraph = GD::Graph::lines->new(1200, 600);

$mygraph->set(
    transparent       => '0',
    bgclr             => 'lgray',
    boxclr            => 'white',
    fgclr             => 'white',
    x_label     => 'steps',
    y_label     => '%',
    title       => 'ML Winrate',
    # Draw datasets in 'solid', 'dashed' and 'dotted-dashed' lines
    line_types  => [1, 2, 4],
    # Set the thickness of line
    line_width  => 2,
    # Set colors for datasets
    dclrs       => ['green', 'red', 'cyan'],
) or warn $mygraph->error;

$mygraph->set_legend_font(['verdana', 'arial', GD::gdMediumBoldFont], 24);
$mygraph->set_title_font(['verdana', 'arial', GD::gdMediumBoldFont], 32);

$mygraph->set_legend('Wins', 'Los', 'Draw');

my $myimage = $mygraph->plot(\@pldata) or die $mygraph->error;

open(IMG, ">:unix", '/data/ml/plot.png') or &Error($!);
binmode IMG;
print IMG $myimage->png;
close(IMG);




