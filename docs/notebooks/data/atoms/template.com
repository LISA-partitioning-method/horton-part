%chk=atom.chk
%nproc=1
%mem=1GB
#PBEPBE/6-31+g** scf=tight

fubar

${charge} ${mult}
${element}    0.0000    0.0000    0.0000
