from itertools import chain
from typing import Self, Union

import pydantic
from quasar_models.modeling import BaseModel
from quasar_typing.astropy import CompoundModelType
from quasar_utils.setup import Info

from .balmer import SerializedBalmerModel
from .gaussian import SerializedGaussianModel
from .host import SerializedHostGalaxyModel
from .iron import SerializedIronModel
from .powerlaw import SerializedPowerLawModel
from .vprofilecopy import SerializedVProfileCopyModel


class SerializedCompoundModel(pydantic.BaseModel):
    pl: list[SerializedPowerLawModel] | None
    fe: list[SerializedIronModel] | None
    ba: list[SerializedBalmerModel] | None
    hg: list[SerializedHostGalaxyModel] | None
    em: list[SerializedGaussianModel | SerializedVProfileCopyModel] | None

    @pydantic.field_validator("pl", "fe", "ba", "hg", "em", mode="after")
    @classmethod
    def validate_fields(cls, field: list | None) -> list | None:
        return field or None

    ###

    @classmethod
    def from_compound_model(
        cls, 
        compound_model: Union[BaseModel, CompoundModelType[BaseModel]],
        info: Info,
    ) -> Self:
        submodels: list[BaseModel]
        if compound_model.n_submodels == 1:
            submodels = [compound_model]
        else:
            submodels = sorted(compound_model, key=lambda m: m.sorting_key)

        fields = {
            "pl": [],
            "fe": [],
            "ba": [],
            "hg": [],
            "em": [],
        }
        for submodel in submodels:
            match submodel.model_type:
                case "pl":
                    seri_cls = SerializedPowerLawModel
                case "fe":
                    seri_cls = SerializedIronModel
                case "ba":
                    seri_cls = SerializedBalmerModel
                case "hg":
                    seri_cls = SerializedHostGalaxyModel
                case "em":
                    seri_cls = SerializedVProfileCopyModel \
                        if hasattr(submodel, "n_profiles") \
                        else SerializedGaussianModel
            
            fields[submodel.model_type].append(
                seri_cls.from_model(submodel, info),
            )

        return cls(**{key: value or None for key, value in fields.items()})

    def to_model(self, info: Info) -> Union[BaseModel, CompoundModelType[BaseModel]]:
        submodels: list[BaseModel] = sorted(
            [
                m.to_model(info)
                for m in chain(
                    self.pl or [],
                    self.fe or [],
                    self.ba or [],
                    self.hg or [],
                    self.em or [],
                )
            ],
            key=lambda m: m.sorting_key,
        )
        return sum(submodels[1:], start=submodels[0])
