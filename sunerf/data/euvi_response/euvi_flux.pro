function euvi_flux, Temp, fits_header, wl, ahead=ahead, behind=behind, $
                   exposure=exposure, photons=photons, 		$
		   channel=channel, filter=filter, 			$
		   use_sre_file=use_sre_file,			$
		   apec=apec,feldman=feldman,all_lines=all_lines,$
                   version=version,sre_file=sre_file,		$ ; Output
		   prog_version=prog_version			  ; Output
;+
;NAME:		euvi_flux
;PURPOSE:	Return the predicted number of DN as a function of 
;		coronal temperature, wavelength (channel) and filter 
;		for EUVI A and EUVI B.  There are four chanels, each with four
;		filters (16 combinations).
;
;		The results assume a volume EM = 1.e44 cm^3
;
;		euvi_flux uses information from the appropriate SRE file
;		as read by the function subroutine rd_euvi_sre.  Results
;		using Chianti (with and without /all_lines) and Apec
;		are available for two assumptions of elemental abundances.
;
;		If the /photons switch is set, return number of photons
;		that are presented to the CCD.  In this case, the amplifier,
;		CCD temperature are not used, as the number of photons is
;		independent of the system gain.
;
;
;CALLING SEQUENCE:
;  F = euvi_flux(Temp,fits_head)		   ; Defaults to Chianti, Fludra
;  F = euvi_flux(Temp,fits_head,/apec)	   ; Use Apec calculation
;  F = euvi_flux(Temp,fits_head,/Feldman)   ; Use Feldman abundances
;  F = euvi_flux(Temp,fits_head,All_lines=0); Don't use all lines in Chianti
;  F = euvi_flux(Temp,fits_head,/phot)		   ; Return photons (instead of DN)
;  F = euvi_flux(Temp,fits_head,use_sre_file=sre_file)	; Provide SRE filename
;
; The following examples illustrate how to use euvi_flux in isolation of
; a specific EUVI FITS header:
; 

;  F = euvi_flux(Temp,wl,/ahead)  ; Specify wavelength (channel) as a parameter
;  F = euvi_flux(Temp,ab,wl)  ; Specify spacecraft (A/B) and wavelength as parameters
;  F = euvi_flux(Temp,channel=channel,filter=filter,/behind) ; Results by
;          specific wavelength (channel) and filter (if omitted "S1" is assumed.)
;  
;
;INPUTS:
;  Temp		= Temperature (K).  May be a scalar or a vector
;
;  fits_header	= If the second parameter is a structure, it is assumed
;			that this is the EUVI FITS header.  
;  ab = 'A' or 'B' or 0 (for 'A') or 1 (for 'B') in place of fits_header, but only as a scalar 
;
;  channel  =  If the third parameter is a scalar or a vector, it is
;	       assumed to be equal to one of the four wavelengths
;	       (171, 195, 284 and 304).   In this case, filter='S1' is
;	       used unless the 'filter' is explicitly specified as a keyword.
;              The channel parameter is given either in strings (i.e.,
;              '171','195','284', and '304') or in numbers (0, 1, 2,
;              and 3, corresponding to '171','195','284', and '304', respectively)  

;
;
;  Either Temp or fits_header [or specacraft] may be vectors.
;  If both are vectors then the returned array is two dimensional.
;  That is, F[i,j] is returned if TEMP=TEMP[i] and fits_header=fits_header[j]. 
;
;OUTPUTS:
;  This function returns DN be default.
;  If /photons is set, return the number of photons that impinge on the CCD.
;
;OPTIONAL INPUT KEYWORDS (if fits_header is NOT specified):
;  All of these parameters must be scalars
;
;  date		= Time in any format.  Reserved to update on-orbit
;                 calibration in case the necessity arises.
;		  NOTE: Ues of date is not currently implemented
;  exposure     = The exposure time in sec    [if not set or specified, Default = 1.0]
;
;OTHER OPTIONAL INPUT KEYWORDS:
;  All of these keywords must be scalars
;
;  photons      = If set, return  photon flux (rather than DN).
;  use_sre_file = If provided, read the specified SRE file.
;		   Must include the entire pathname.  This will override
;		   the FM= keyword and the three atomic keywords (Apec,
;		   Feldman, and all_lines).
;  Apec		= If set, use the Apec calculation instead of Chianti
;  Feldman	= If set, use Feldman abundances instead of Fludra
;  all_lines	= Default is set.  Specify all_lines = 0, to use the
;		  limited line list calculation (only relevant for Chianti)
;
;OPTIONAL OUTPUT KEYWORDS:
;  version      = version number of the [First] SRE file
;  sre_file	= Name of the [first] SRE file from which data is extracted.
;		  If fits_header is supplied, it is possible that multiple
;		  FM instruments are included, and thus multiple SRE file will
;		  be used.  However, only the first file name is returned.
;  prog_version	= A structure containing the version of this program and the
;		  version of the euvi_gain program
;
;MODIFICATION HISTORY:
; 11-Apr-2008, N. V. Nitta, V1.00  Written (first release), based on
;              sxi_flux.pro.  Things are much simpler here.

;-

prog_version = {euvi_flux:1.00}	
@euvi_sre_common
;common euvi_sre_db, euvi_sre_common, Filename_common, Full_filename_common

; if no parameters are present, assume information mode

if n_params() lt 2 then begin
  print,format="(75('-'))"
  doc_library,'euvi_flux'
  print,format="(75('-'))"
  return,-1
endif



;----------------------------------------------------------------------------
;  ****  Step 1:  Set up values optional input parameters:
;----------------------------------------------------------------------------

n_Temp = n_elements(Temp)
if n_Temp eq 0 then message,'Temp variable is undefined'
n_Chan = n_elements(fits_header)
szt=size(fits_header, /type)

; qcase = 1 ==> Fits_header = structure
; qcase = 2 ==> Channel and /ahead or /behind keyword
; qcase = 3 ==> Spacecraft (A or B) and Channel provided as a string ('171','195','284','304')
; or number (0,1,2,3)

if szt eq 8 then begin ; structure which must be the EUVI header structure 
   qcase=1 
   goto, st2
endif

if n_params() ge 3 then begin  ; this must be qcase=3 and the second parameter(fits_header) must be A/B
   if szt lt 1 or (szt gt 5 and szt ne 7) then begin
      print,'the second parameter, if not an EUVI header structure, must be a string or '
      print,' non-complex number to specify spacecraft ("A" or "B" or 0 ("A") or 1 ("B"))'
      return,-1
   endif
   scab=intarr[n_Chan] ; expect n_Chan to be 1
   if szt eq 7 then begin
      for i=0, nChan-1 do begin
         if fits_header[i] eq 'B' then scab[i]=1 else scab[i]=0
      endfor
   endif else begin
      for i=0, nChan-1 do begin
         if fits_header[i] eq 1 then scab[i]=1 else scab[i]=0
      endfor
   endelse
;    Note that the vector information is not used.  Assume AB (in
;    place of fits_header) is a scalar   
   qcase=3
   n_Chan=n_elements(wl)
   cchan=euvi_wavelnth_chan(wl)   
 endif else begin  ;  n_params()=2 then the second para should be channel (wavelength) needing /ahead or /behind
   if (not keyword_set(ahead) and not keyword_set(behind)) or $
    (keyword_set(ahead) and keyword_set(behind)) then begin
      print,'Two parameters specified (temperature and channel).  You should set either /ahead or /behind.'
      print,'/ahead is assumed'
      scab=replicate(0, n_Chan)
      goto, ab_assigned 
   endif
   if keyword_set(ahead) then scab=replicate(0, n_Chan) else scab=replicate(1, n_Chan)

ab_assigned:

   qcase=2
   cchan=euvi_wavelnth_chan(fits_header)
endelse

if not keyword_set(filter) then cfilt=replicate(1, n_Chan) else begin ; assume filter='S1'
   if n_elements(filter) ge 1 then begin
      if size(filter,/type) ne 7 then cfilt=replicate(filter[0], n_Chan) else $  ; must be 0,1,2 or 3
         cfilt=replicate(euvi_filter_n(filter[0]), n_Chan)
   endif
endelse

st2:
;----------------------------------------------------------------------------
;  ****  Step 2:  Loop on channel counter (e.g. FITS header)	
;----------------------------------------------------------------------------

outarr = fltarr(n_elements(Temp), n_Chan)

if qcase eq 1 then begin
   sre = rd_euvi_sre(Fits_header[0], date=date, 		$  ; don't assume mixture of A and B
               use_sre_file=use_sre_file, apec=apec,           	$
                Feldman=Feldman, All_lines = All_lines,         $
                debug=debug,                                    $
                sre_filename=sre_file1)                         ;  Output
   cchan=euvi_wavelnth_chan(Fits_header)
   cfilt=euvi_filter_n(Fits_header)
endif else $
             sre=rd_euvi_sre(ab=scab[0], date=date, 		$  ; don't assume mixture of A and B
               use_sre_file=use_sre_file, apec=apec,           	$
                Feldman=Feldman, All_lines = All_lines,         $
                debug=debug,                                    $
                sre_filename=sre_file1)

 
sre_file = sre_file1
version = sre.version

for i=0,n_Chan-1 do begin

  if keyword_set(photons) then begin
    outarr[0,i] = dspline(sre.Temp,sre.phot[*, cfilt[i], cchan[i]], Temp,interp=0)
  endif else begin
    qgain = 15.   ; hard wired number for now
    outarr[0,i] = dspline(sre.Temp,sre.elec[*, cfilt[i], cchan[i]], Temp,interp=0)/qgain
  endelse

; *** Apply the exposure correction		***

  if qcase eq 1 and keyword_set(exposure) then qExp = Fits_header[i].exptime else $
    if n_elements(exposure) ne 0 then qExp = exposure else qExp = 1.0
  outarr[*,i] = outarr[*,i] * qExp
endfor 				 

return, outarr
end
