# this is a collection of utilities that would
# probably be needed by many modules in this package

import yaml
from datetime import datetime, timezone
import dateutil.parser
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time


def get_username_password(service, filename="passwords.yml"):
    """
    Load the credentials from the passwords.yml file,
    and return a tuple with the username and password
    for the required service.

    Parameters
    ----------
    service: scalar str
        The service for which the credentials need to be loaded.
        Must be a top-level item in the YAML file.

    Returns
    -------
    credentials: 2-tuple of strings
        A tuple that contains the username and password.

    """
    with open(filename) as file:
        serv_dict = yaml.safe_load(file)
        credentials = serv_dict.get(service, None)
        if credentials is None:
            raise KeyError(f'Cannot find credentials for service "{service}". ')

        values = []
        if "username" in credentials:
            values.append(credentials.get("username"))
        if "password" in credentials:
            values.append(credentials.get("password"))

        return values


def ra2deg(ra):
    """
    Convert the input right ascension into a float of decimal degrees.
    The input can be a string (with hour angle units) or a float (degree units!).

    Parameters
    ----------
    ra: scalar float or str
        Input RA (right ascension).
        Can be given in decimal degrees or in sexagesimal string (in hours!)
        Example 1: 271.3
        Example 2: 18:23:21.1

    Returns
    -------
    ra: scalar float
        The RA as a float, in decimal degrees

    """
    if type(ra) == str:
        c = SkyCoord(ra=ra, dec=0, unit=(u.hourangle, u.degree))
        ra = c.ra.value  # output in degrees
    else:
        ra = float(ra)

    if not 0.0 < ra < 360.0:
        raise ValueError(f"Value of RA ({ra}) is outside range (0 -> 360).")

    return ra


def dec2deg(dec):
    """
    Convert the input right ascension into a float of decimal degrees.
    The input can be a string (with hour angle units) or a float (degree units!).

    Parameters
    ----------
    dec: scalar float or str
        Input declination.
        Can be given in decimal degrees or in sexagesimal string (in degrees as well)
        Example 1: +33.21 (northern hemisphere)
        Example 2: -22.56 (southern hemisphere)
        Example 3: +12.34.56.7

    Returns
    -------
    dec: scalar float
        The declination as a float, in decimal degrees

    """
    if type(dec) == str:
        c = SkyCoord(ra=0, dec=dec, unit=(u.degree, u.degree))
        dec = c.dec.value  # output in degrees
    else:
        dec = float(dec)

    if not -90.0 < dec < 90.0:
        raise ValueError(f"Value of dec ({dec}) is outside range (-90 -> +90).")

    return dec


def date2jd(date):
    """
    Parse a string or datetime object into a Julian Date (JD) float.
    If string, will parse using dateutil.parser.parse.
    If datetime, will convert to UTC or add that timezone if is naive.
    If given as float, will just return it as a float.

    Parameters
    ----------
    date: float or string or datetime
        The input date or datetime object.

    Returns
    -------
    jd: scalar float
        The Julian Date associated with the input date.

    """
    if isinstance(date, datetime):
        t = date
    elif isinstance(date, str):
        t = dateutil.parser.parse(date)
    else:
        return float(date)

    if t.tzinfo is None:  # naive datetime (no timezone)
        # turn a naive datetime into a UTC datetime
        t = t.replace(tzinfo=timezone.utc)
    else:  # non naive (has timezone)
        t = t.astimezone(timezone.utc)

    return Time(t).jd


if __name__ == "__main__":
    # test that loading credentials works:
    credentials = get_username_password("ztf")
    print(f"User: {credentials[0]} | Pass: {credentials[1]}")

    # test that unit conversions work:
    ra = "18:30:12.3"
    print(f'RA string: "{ra}" | float {ra2deg(ra)} degrees.')
    dec = "+31:23:56.1"
    print(f'Dec string: "{dec}" | float {dec2deg(dec)} degrees.')

    # test the convertion to JD
    jd = 2459717.5
    print(f"time in JD: {jd} | converted to JD: {date2jd(jd)}")
    date = "2022-05-19"
    print(f"time as string: {date} | converted to JD: {date2jd(date)}")
    time = datetime(2022, 5, 19)
    print(f"time as datetime: {time} | converted to JD: {date2jd(time)}")
    tz = datetime.now(timezone.utc).astimezone().tzinfo
    time = datetime(2022, 5, 19, tzinfo=tz)
    print(f"time as datetime: {time} | converted to JD: {date2jd(time)}")
