function	rd_euvi_sre, Fits_header, ab=ab, date=date, 	$
		use_sre_file=use_sre_file, apec=apec, 		$
		Feldman=Feldman, All_lines = All_lines, 	$
                debug=debug,			 		$
                sre_filename=sre_file1				; Output

;+
; $Id: rd_euvi_sre.pro,v 1.2 2012/10/23 19:55:57 nathan Exp $
;
;NAME:		rd_euvi_sre
;PURPOSE:	Read the appropriate SRE file that contains the
;		emission response function for the EUVI on STEREO A and B
;
;		The SRE data structure is returned.
;
;		This routine is called by euvi_flux
;
;CALLING SEQUENCE:
;
; sre = rd_euvi_sre(Fits_header)
; sre = rd_euvi_sre(ab=0, date=date)
; sre = rd_euvi_sre(Fits_header, use_sre_file = sre_filename); Provide filename
; sre = rd_euvi_sre(Fits_header,/apec,/Feldman,/all_lines)
; sre = rd_euvi_sre(Fits_header, sre_file=sre_filename)	; Return filename used
;
; Either an EUVI Fits_header or the AB= keyword must be supplied 
;
; The date= keyword is included for future possible use, but is
; not currently used.
;
; Note: All input parameters are assumed to be scalars
;
;INPUT PARAMETERS:
; Fits_header		= EUVI FITS file header from which the
;			  spacecraft information (Ahead or Behind) is 
;                         extracted, so is date (in case the
;                         instrument response function
;			  changes with time)
;
;OPTIONAL INPUT KEYWORDS:
; AB			= Spacecraft A or B.  This can be specified
;                         either as a string ('A' or 'B') 
;                         or as a number (0 for 'A' and 1 for 'B')
;                         AB overrides the value found
;			  in Fits_header if Fits_header is present
;
; Date			= Date of the data.  Date ovverides the value found
;			  in Fits_header if Fits_header is present
;
; use_sre_file		= If present, read the specified file regardless
;			  of all other keywords
;
; Apec			= If set, use the Apec calucation instead of Chianti
; Feldman		= If set, use Feldman abundances instead of Fludra
; all_lines		= Use all_lines = 1 for  *_sre_chianti2_*geny file
;			  Use all_lines = 0 for  *_sre_chianti1_*geny file
;			  Default is /all_lines
;
;OPTIONAL OUTPUT KEYWORDS:
; sre_filename		= The full path name of the SRE file that was read.
;
;PROCEDURE:
; Unless USE_SRE_FILE input keyword is specified, rd_euvi_sre will
; assume that target directory for the sre files is defined by 
; $EUVI_RESPONSE.  If file name is passed in explicitly via use_sre_file, 
; the full file path name must be explicitly provided.
;
; After the first call, will check to see if the file name of the
; file to be read is different than the current file.  If not, no
; read is performed and the contents of the previously read file 
; are returned.
;
;RESTRICTIONS:
; All input parameters and keywords must be scalars.
;
;COMMON BLOCKS:
; euvi_sre_common	Contains the contents of the SRE file.
;
;HISTORY: 
; 11-Apr-2007, N. V. Nitta, Written on the basis of rd_euvi_sxi.pro
; 21-Sep-2012, N. V. Nitta, Fixed a bug that prevented the
;                  "use_sre_file" keyword from working as expected
;-

ab01=['ahead','behind']

@euvi_sre_common
;common euvi_sre_db, euvi_sre_common, Filename_common, Full_Filename_common

target_dir = '$EUVI_RESPONSE'	; Default target directory for SRE files

; --- Step 1:  Create the target file name ------------------------


if keyword_set(use_sre_file) then begin
  target_filename = use_sre_file
  filename=file_break(target_filename)
  cab=strmid(filename,0,strpos(filename,'_'))
     print,'filename = ',filename
  infile=target_filename
  goto, ab_finished
endif else begin
   if n_elements(AB) eq 1 then begin
      if size(AB, /type) eq 7 then begin
         if AB ne 'A' and AB ne 'B' then begin
            print,'keyword AB has to be "A" or "B" when it is a string'
            return, -1
         endif
         if AB eq 'B' then qab=1 else qab=0      
      endif else begin
         if size(AB, /type) lt 1 or size(AB,/type) gt 5 then begin
            print,'keyword AB has to be a byt, int, flt, or dbl'
            return, -1
         endif                    
         qab=fix(ab)
         if qab ne 0 and qab ne 1 then begin
            print,'keyword AB has to be 0 or 1 when it is a number'
            return, -1
         endif
      endelse
      goto, ab_determined        
   endif
   if n_elements(Fits_header) ne 1 then begin
      print,'Fits_header must be a scalar'
      return, -1
   endif
   if size(Fits_header,/type) ne 8 then begin
      print,'Fits_header must be an EUVI header structure when AB= is not set'
      return, -1
   endif
   if tag_exist(Fits_header, 'obsrvtry') ne 1 then begin
      print,'Fits_header has to have a tag called "obsrvtry"'
      return, -1
   endif
   hab=strmid(fits_header.obsrvtry, strlen(fits_header.obsrvtry)-1, 1) 
   if hab ne 'A' and hab ne 'B' then begin
      print,'Fits_header.obsrvtry mustendd with either "A" or "B"'
      return, -1
   endif
   if hab eq 'A' then qab=0 else qab=1
endelse


; ****************************************************************
; JRL: The date keyword is not currently used 
;
; If the response function of the EUVI changes with time, it may
; be necessary to uncomment the following lines and to change
; the algorithm to determine the sre filename.
;
;  if keyword_set(date) then qdate = date else $
;		          qdate = gt_euvi_params(Fits_header, /date)
; ****************************************************************


ab_determined:

cab=ab01[qab]


code = ['chianti','apec']
if n_elements(all_lines) eq 0 then q_all = 1 else q_all = all_lines
if keyword_set(apec) then alines = ['', ''] else alines = ['1','2']
abun = ['fludra','feldman']
target_filename = string(cab+'_sre_',		$
			code[keyword_set(apec)],		$
			alines[keyword_set(q_all)]+'_',		$
			abun[keyword_set(Feldman)],		$
			'_mazzotta_*.geny')
  target_filename = concat_dir( target_dir, target_filename )

; --- Step 2:  Search for the target file name ------------------------

files = file_search(target_filename,count=count,/expand_environ)

if count eq 0 then begin
  message,'No sre files found: ', /cont
  print,' ' + target_filename
  return, -1
endif

;  More than one version of the file may exist.
;
;  The following statement chooses the "highest" version (assuming
;  that the conversion of version = 001, 002, ... is followed).

infile = files[count-1]

; --- Step 3:  Read the file if this is the first call ----------------
;              or read the file if a different file is requested

ab_finished:



break_file, infile, disk_log, dir, filnam, ext
if n_elements(Filename_common) eq 0 then Filename_common = ''
if filnam ne Filename_common then begin
  restgenx, file=infile, euvi_sre_common
  Filename_common=filnam		; Set common variables
  Full_Filename_common = infile
debug = 0
  if keyword_set( debug ) then print,'** Reading = ',filnam
endif

sre = euvi_sre_common
sre_file1 = Full_Filename_common

return, sre
end
