#!/bin/bash

# make sure g16 and formchk are available before running this script.

MISSING=0
if ! which g16 &>/dev/null; then echo "g16 binary not found."; MISSING=1; fi
if ! which formchk &>/dev/null; then echo "formchk binary not found."; MISSING=1; fi
if [ $MISSING -eq 1 ]; then echo "The required programs are not present on your system. Giving up."; exit -1; fi

function do_atom {
    echo "Computing in ${1}"
    cd ${1}
    if [ -e atom.out ]; then
        echo "Output file present in ${1}, not recomputing."
    else
        g16 atom.in > atom.out
        RETCODE=$?
        if [ $RETCODE == 0 ]; then
            formchk atom.chk atom.fchk
            rm -f atom.out.failed
        else
            # Rename the output of the failed job such that it gets recomputed
            # when the run script is executed again.
            mv atom.out atom.out.failed
        fi
        rm atom.chk
    fi
    cd -
}

for ATOMDIR in [01][0-9][0-9]_*_[01][0-9][0-9]_q[-+][0-9][0-9]/mult[0-9][0-9]; do
    do_atom ${ATOMDIR}
done
