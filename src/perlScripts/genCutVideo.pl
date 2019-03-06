#!/usr/bin/perl -lw

use strict;

###
### GLOBAL FUNCTIONS ###
###

sub ReadCmdLine;
sub GeneratePgmFiles;
sub RemovePgmFiles;
sub GenerateFullMovie;
sub DeriveDifferences;
sub ApplyDiffToFrames;
sub BuildHighlightMovie;

###
### GLOBAL VARIABLES ###
###

my $sourceDir = "";
my $fullVideoOut = "fullVideo.gif";
my $maskVideoOut = "maskVideo.gif";
my $nonMaskVideoOut = "nonMaskVideo.git"
my $sourceMaskFile = "";
my $newSourceMask = "";

my @frameFiles = ();
my @diff;
my $randNum;

###
### MAIN ### 
###

srand(time());
$randNum = rand()*100%100;

ReadCmdLine();
print "Generating PGM files";
GeneratePgmFiles();
print "Generating Full Movie";
GenerateFullMovie();
print "Deriving Difference";
DeriveDifferences();
print "Generating New Frames";
ApplyDiffToFrames();
print "Building Highlight Movie";
BuildHighlightMovie();
print "Removing all files";
RemovePgmFiles();
print "Done!";

###
### FUNCTIONS ###
###

sub BuildHighlightMovie {
    foreach my $file (@frameFiles){
        `convert tmp\_$randNum/$file\_diff.pgm tmp\_$randNum/$file\_diff.png`;
    }

    `convert -set delay 3 -colorspace RGB -colors 256 -dispose 1 -loop 0 -scale 50% tmp\_$randNum/*frame*_diff.png $highLightVideoOut`    
}

sub ApplyDiffToFrames {

    foreach my $file (@frameFiles){
        my $newFileContent = "";
        my $allContent = "";
        open(IN,"tmp\_$randNum/$file.pgm") or die "ERROR: Unable to open file tmp\_$randNum/$file.pgm\n";
        <IN>;
        my $line1 = "P3\n";
        my $line2 = <IN>;
        my $line3 = <IN>;
        while(<IN>){
            chomp;
            $allContent .= " " . $_;
        }
        close(IN);
        $allContent =~ s/^\s+//;
        my @allContent = split(/\s+/,$allContent);
        my $len1 = @allContent;
        my $len2 = @diff;
        if( $len1 != $len2 ){
            die "ERROR: The diff and allContent arrays don't match in size. ($len1,$len2)\n";
        }
        for( my $i = 0; $i < @allContent; $i++ ){
            if( $diff[$i] == 0 ){
                $newFileContent .= " $allContent[$i] $allContent[$i] $allContent[$i]";
            } elsif ( $diff[$i] == 2 ){
                $newFileContent .= " 255 $allContent[$i] $allContent[$i]";
            } elsif ( $diff[$i] == 3 ){
                $newFileContent .= " $allContent[$i] 255 $allContent[$i]";
            } elsif ( $diff[$i] == 4 ){
                $newFileContent .= " $allContent[$i] 255 255";
            } else {
                die "ERROR: Unrecognized diff value '$diff[$i]'\n";
            }
        }
        $newFileContent =~ s/^\s+//;
        open(OUT,"> tmp\_$randNum/$file\_diff.pgm") or die "ERROR: Unable to open the file tmp\_$randNum/$file\_diff.pgm";
        print OUT "$line1$line2$line3$newFileContent";
        close(OUT);
    }    
}

sub DeriveDifferences {
    my @source = ();
    my @predict = ();
    @diff = ();
    my $content = "";

    open(IN,"tmp\_$randNum/$newSourceMask") or die "ERROR: Unable to open tmp\_$randNum/$newSourceMask";
    my $line1 = <IN>;
    my $line2 = <IN>;
    my $line3 = <IN>;
    while(<IN>){
        chomp;
        $content .= " " . $_;
    }
    close(IN);
    $content =~ s/^\s+//;
    @source = split(/\s+/,$content);

    open(IN,"tmp\_$randNum/$newPredictMask") or die "ERROR: Unable to open tmp\_$randNum/$newPredictMask";
    <IN>;
    <IN>;
    <IN>;
    $content = "";
    while(<IN>){
        chomp;
        $content .= " " . $_;
    }
    close(IN);
    $content =~ s/^\s+//;
    @predict = split(/\s+/,$content);

    my $sourceLen = @source;
    my $predictLen = @predict;
    if( $sourceLen != $predictLen ){
        RemovePgmFiles();
        die "ERROR: The two mask files don't carray the same number of pixels\n";
    }
    for( my $i = 0; $i < @source; $i++ ){
        if( $source[$i] == 2 or $predict[$i] == 2 ){
            if( $source[$i] == 2 and $predict[$i] == 2 ){
                push @diff, 2;
            } elsif ($source[$i] == 2 ){
                push @diff, 3;
            } else {
                push @diff, 4;
            }
        } else {
            push @diff, 0;
        }
    }
}

sub GenerateFullMovie {
    foreach my $file (@frameFiles){
        `convert tmp\_$randNum/$file.pgm tmp\_$randNum/$file.png`;
    }
    `convert -set delay 3 -colorspace GRAY -colors 256 -dispose 1 -loop 0 -scale 50% tmp\_$randNum/*frame*.png $fullVideoOut`
}

sub GeneratePgmFiles {
    # Build a temporary directory
    `mkdir tmp_$randNum`;
    my $built = 0;
    open(LS,"ls -d tmp\_$randNum/ |");
    while(<LS>){
        $built = 1;
    }
    close(LS);
    if( $built == 0 ){
        die "ERROR: Unable to create a temporary work area directory here.\n";
    }

    # Copy over the frames
    open(LS,"ls $sourceDir/*frame*\.png |");
    while(<LS>){
        chomp;
        $_ =~ s/.*\/([^\/]+)$/$1/;
        $_ =~ s/\.png$//;
        push @frameFiles, $_;
    }
    close(LS);
    if( @frameFiles != 100 ){
        die "ERROR: Expected to find 100 frames.\n";
    }

    # Convert the mask files
    `convert $sourceMaskFile -colorspace gray -compress none tmp\_$randNum/$newSourceMask`;
    `convert $predictMaskFile -colorspace gray -compress none tmp\_$randNum/$newPredictMask`;
    
    # Convert the frame files
    for( my $i = 0; $i < @frameFiles; $i++ ){
        `convert $sourceDir/$frameFiles[$i].png -colorspace gray -compress none tmp\_$randNum/$frameFiles[$i].pgm`; 
    }

    # Done making files to work with
}

sub RemovePgmFiles {
    `rm -R -f tmp\_$randNum`;
}

sub ReadCmdLine {
    for( my $i = 0; $i < @ARGV; $i++ ){
        if( $ARGV[$i] eq "-d" ){
            $i++;
            $sourceDir = $ARGV[$i];
        } elsif( $ARGV[$i] eq "-f" ){
            $i++;
            $fullVideoOut = $ARGV[$i];
        } elsif( $ARGV[$i] eq "-m" ){
            $i++;
            $maskVideoOut = $ARGV[$i];
        } elsif( $ARGV[$i] eq "-n" ){
            $i++;
            $nonMaskVideoOut = $ARGV[$i];
        } elsif( $ARGV[$i] eq "-s" ){
            $i++;
            $sourceMaskFile = $ARGV[$i];
        }
    }

my $fullVideoOut = "fullVideo.gif";
my $maskVideoOut = "maskVideo.gif";
my $nonMaskVideoOut = "nonMaskVideo.git"

    if( $sourceDir eq "" or $sourceMaskFile eq "" ){
        die <<EOF;
ERROR: Missing command line arguments.
./genCutVideo.pl -d <directory containing all frame png files>
    -s <correct mask png file>
    (optional: 
        -f <output gif film filename [def. fullVideo.gif]>
        -m <output gif mask only film filename [def. maskVideo.gif]>
        -n <output gif non-mask only film filename [def. nonMaskVideo.gif]>
    )
>>> WARNING <<<
This script generates a temporary directory (tmp_[rand num]), places
files there, and finally remove all of it.
EOF
    }
    if( $fullVideoOut !~ /\.gif$/ or $maskVideoOut !~ /\.gif$/ or $nonMaskVideoOut !~ /\.gif$/ ){
        die "ERROR: Please specify output film filenames ending with '.gif'\n";
    }
    if( $sourceMaskFile !~ /\.png/ ){
        die "ERROR: Please provide png mask files\n";
    }

    $sourceDir =~ s/\/$//;
    $newSourceMask = $sourceMaskFile;
    $newSourceMask =~ s/^.*\/([^\/]+)\.png$/$1\_source.pgm/;
    $newPredictMask = $predictMaskFile;
    $newPredictMask =~ s/^.*\/([^\/]+)\.png$/$1\_predict.pgm/;
}

