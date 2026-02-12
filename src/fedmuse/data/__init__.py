# Import dataset builders so they register into fedmuse.registry on import.
from .pacs import build_pacs_domains  # noqa: F401
from .office_home import build_office_home_domains  # noqa: F401
from .domainnet import build_domainnet_domains  # noqa: F401
