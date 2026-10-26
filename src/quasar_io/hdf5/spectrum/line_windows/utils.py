from quasar_decrypt.lines import LWindow


def clear_lwindow(lwindow: LWindow) -> None:
    """Clears the fields of an LWindow instance."""
    lwindow.names.clear()
    lwindow.linetypes.clear()
    lwindow.complexes.clear()
    lwindow.waves.clear()
    lwindow.n_maxs.clear()
    lwindow.needs_line.clear()

    lwindow.strength_bounds.clear()
    lwindow.fwhm_v_bounds.clear()
    lwindow.v_off_bounds.clear()

    lwindow.is_copy_of.clear()
    lwindow.is_multiplet_of.clear()

    lwindow.scale_init.clear()
    lwindow.scale_bounds.clear()
    lwindow.scale_fixed.clear()

    lwindow.copies_to.clear()

    lwindow.i_bounds.clear()
    lwindow.blacklist.clear()
    lwindow._blacklist.clear()
    
    lwindow.neighbours = (None, None)
    lwindow.model = None
    lwindow.fit = None
    lwindow.fit_info = None
    lwindow.fits.clear()
    lwindow.fit_infos.clear()

    lwindow.cropped.clear()
    lwindow.bootstrapper = None
    lwindow.error_result = None
    lwindow._y_em_contrib[:] = 0.0