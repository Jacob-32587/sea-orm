use super::compound::{BelongsTo, BelongsToCardinality, HasMany, HasOne};
use crate::{ActiveModelTrait, DbErr, EntityTrait, ModelTrait, TryIntoModel};
use core::ops::{Index, IndexMut};

/// State carried by a `belongs_to` field on an
/// [`ActiveModelEx`](crate::EntityTrait::ActiveModelEx). Mirrors the
/// `NotSet` / `Set` shape of [`ActiveValue`](crate::ActiveValue) but for a
/// related model.
///
/// Unstable: nested-`ActiveModel` relation mutation is exempt from semver — the
/// semantics of setting or removing related records may change in a minor (2.x) release.
#[derive_where::derive_where(Debug, Clone, PartialEq, Eq; <T as BelongsToCardinality>::Set)]
#[derive(Default)]
pub enum ActiveBelongsTo<T>
where
    T: BelongsToCardinality,
{
    /// Field is absent; the related model is left as-is on save.
    #[default]
    NotSet,
    /// Set the related ActiveModel on save.
    Set(<T as BelongsToCardinality>::Set),
}

/// State carried by a `has_one` field on an
/// [`ActiveModelEx`](crate::EntityTrait::ActiveModelEx).
///
/// Unstable: nested `ActiveModel` relation mutation is exempt from semver - the
/// semantics of setting or removing related records may change in a minor (2.x) release.
#[derive_where::derive_where(Debug, Clone, PartialEq, Eq; E::ActiveModelEx)]
#[derive(Default)]
pub enum ActiveHasOne<E>
where
    E: EntityTrait,
{
    /// Field is absent; the related model is left as-is on save.
    #[default]
    NotSet,
    /// Set or clear the related ActiveModel on save.
    Set(Option<Box<E::ActiveModelEx>>),
}

/// State carried by a `has_many` (or many-to-many) field on an
/// [`ActiveModelEx`](crate::EntityTrait::ActiveModelEx). Chooses between
/// "leave alone", "additive write", and "destructive replace" semantics.
///
/// Unstable: nested-`ActiveModel` relation mutation is exempt from semver - the
/// semantics of replacing or removing related records may change in a minor (2.x) release.
#[derive_where::derive_where(Debug, Clone, PartialEq, Eq; E::ActiveModelEx)]
#[derive(Default)]
#[non_exhaustive]
pub enum ActiveHasMany<E: EntityTrait> {
    /// Field is absent; existing related models are left as-is on save.
    #[default]
    NotSet,
    /// Persist exactly this list of related models, deleting any existing
    /// children that are not in the list.
    Replace(Vec<E::ActiveModelEx>),
    /// Persist these related models alongside any existing children; never
    /// deletes.
    Append(Vec<E::ActiveModelEx>),

    /// Persists the related model alongside any existing children
    Mutate(Mutation<E>),
}

/// Container that holds models to save and delete
#[derive_where::derive_where(Debug, Clone, PartialEq, Default, Eq; E::ActiveModelEx)]
pub struct Mutation<E: EntityTrait> {
    save: Vec<E::ActiveModelEx>,
    delete: Vec<E::ActiveModelEx>,
}

impl<E: EntityTrait> Mutation<E> {
    /// Create an empty container
    pub fn new() -> Self {
        Self {
            save: vec![],
            delete: vec![],
        }
    }

    /// Create a container with models to delete and save
    pub fn from_save_and_delete(
        save: Vec<E::ActiveModelEx>,
        delete: Vec<E::ActiveModelEx>,
    ) -> Self {
        Self { save, delete }
    }

    /// Create a container with models to save
    pub fn from_save(save: Vec<E::ActiveModelEx>) -> Self {
        Self {
            save,
            delete: vec![],
        }
    }

    /// Create a container with models to delete
    pub fn from_delete(save: Vec<E::ActiveModelEx>) -> Self {
        Self {
            save,
            delete: vec![],
        }
    }

    /// Get a slice to models that will be saved
    pub fn save_as_slice(&self) -> &[E::ActiveModelEx] {
        self.save.as_slice()
    }

    /// Get a mutable vector to models that will be saved
    pub fn save_as_mut_vec(&mut self) -> &mut Vec<E::ActiveModelEx> {
        &mut self.save
    }

    /// Get a mutable vector to models that will be deleted
    pub fn delete_as_mut_vec(&mut self) -> &mut Vec<E::ActiveModelEx> {
        &mut self.delete
    }

    /// Get a slice to models that will be deleted
    pub fn delete_as_slice(&self) -> &[E::ActiveModelEx] {
        self.delete.as_slice()
    }

    /// Consume [self] and produce a single [Vec] of all
    /// models in the container
    pub fn into_vec(mut self) -> Vec<E::ActiveModelEx> {
        self.save.extend(self.delete);
        self.save
    }

    /// Consume [self] and produce a single [Vec] of models
    /// to be saved
    pub fn into_save_vec(self) -> Vec<E::ActiveModelEx> {
        self.save
    }

    /// Iterate over all models in the container
    pub fn iter<I>(&self) -> impl Iterator<Item = &E::ActiveModelEx> {
        self.save.iter().chain(self.delete.iter())
    }
}

impl<E: EntityTrait> IntoIterator for Mutation<E> {
    type Item = E::ActiveModelEx;
    type IntoIter = std::iter::Chain<
        std::vec::IntoIter<E::ActiveModelEx>,
        std::vec::IntoIter<E::ActiveModelEx>,
    >;

    fn into_iter(self) -> Self::IntoIter {
        self.save.into_iter().chain(self.delete)
    }
}

/// Which save operation an [`ActiveModel`](crate::ActiveModelTrait) is about
/// to perform - used by hooks and helpers that need to branch on the kind
/// of write.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ActiveModelAction {
    /// `INSERT`.
    Insert,
    /// `UPDATE`.
    Update,
    /// Insert if the primary key is `NotSet`, otherwise update.
    /// Only meaningful for entities with an auto-increment primary key.
    Save,
}

impl<T> ActiveBelongsTo<T>
where
    T: BelongsToCardinality,
{
    /// Construct an `ActiveBelongsTo::Set`. Accepts a bare active model for a
    /// required relation, or an `Option<active model>` for an optional one.
    pub fn set<M>(model: T::Value<M>) -> Self
    where
        M: Into<<<T as BelongsToCardinality>::Entity as EntityTrait>::ActiveModelEx>,
    {
        Self::Set(T::into_set(model))
    }

    /// Take ownership of this relation state, leaving `NotSet` in place
    pub fn take(&mut self) -> Self {
        std::mem::take(self)
    }

    /// Return true if self is NotSet
    pub fn is_not_set(&self) -> bool {
        matches!(self, Self::NotSet)
    }

    /// Return true if there is a set value
    pub fn is_set(&self) -> bool {
        matches!(self, Self::Set(_))
    }
}

impl<E> ActiveBelongsTo<E>
where
    E: EntityTrait,
{
    /// Get a reference, if set
    pub fn as_ref(&self) -> Option<&E::ActiveModelEx> {
        match self {
            Self::Set(model) => Some(model),
            _ => None,
        }
    }

    /// Get a mutable reference, if set
    #[allow(clippy::should_implement_trait)]
    pub fn as_mut(&mut self) -> Option<&mut E::ActiveModelEx> {
        match self {
            Self::Set(model) => Some(model),
            _ => None,
        }
    }

    /// Return true if the containing model is set and changed
    pub fn is_changed(&self) -> bool {
        match self {
            Self::Set(model) => model.is_changed(),
            _ => false,
        }
    }

    /// Convert into an `Option<ActiveModelEx>`
    pub fn into_option(self) -> Option<E::ActiveModelEx> {
        match self {
            Self::Set(model) => Some(*model),
            Self::NotSet => None,
        }
    }

    /// Convert this back to a `ModelEx` container
    pub fn try_into_model(self) -> Result<BelongsTo<E>, DbErr>
    where
        E::ActiveModelEx: TryIntoModel<E::ModelEx>,
    {
        Ok(match self {
            Self::Set(model) => BelongsTo::Loaded(Box::new((*model).try_into_model()?)),
            Self::NotSet => BelongsTo::Unloaded,
        })
    }
}

impl<E> ActiveBelongsTo<Option<E>>
where
    E: EntityTrait,
{
    /// Get a reference, if set
    pub fn as_ref(&self) -> Option<&E::ActiveModelEx> {
        match self {
            Self::Set(Some(model)) => Some(model),
            _ => None,
        }
    }

    /// Get a mutable reference, if set
    #[allow(clippy::should_implement_trait)]
    pub fn as_mut(&mut self) -> Option<&mut E::ActiveModelEx> {
        match self {
            Self::Set(Some(model)) => Some(model),
            _ => None,
        }
    }

    /// Return true if the containing model is set and changed
    pub fn is_changed(&self) -> bool {
        match self {
            Self::Set(Some(model)) => model.is_changed(),
            Self::Set(None) => true,
            Self::NotSet => false,
        }
    }

    /// Convert into an `Option<ActiveModelEx>`
    pub fn into_option(self) -> Option<E::ActiveModelEx> {
        match self {
            Self::Set(Some(model)) => Some(*model),
            Self::Set(None) | Self::NotSet => None,
        }
    }

    /// Convert this back to a `ModelEx` container
    pub fn try_into_model(self) -> Result<BelongsTo<Option<E>>, DbErr>
    where
        E::ActiveModelEx: TryIntoModel<E::ModelEx>,
    {
        Ok(match self {
            Self::Set(Some(model)) => BelongsTo::Loaded(Some(Box::new((*model).try_into_model()?))),
            Self::Set(None) => BelongsTo::Loaded(None),
            Self::NotSet => BelongsTo::Unloaded,
        })
    }
}

impl<E> ActiveHasOne<E>
where
    E: EntityTrait,
{
    /// Construct an `ActiveHasOne::Set` from an optional related active model.
    pub fn set(model: Option<impl Into<E::ActiveModelEx>>) -> Self {
        Self::Set(model.map(|model| Box::new(model.into())))
    }

    /// Return true if self is NotSet
    pub fn is_not_set(&self) -> bool {
        matches!(self, Self::NotSet)
    }

    /// Return true if there is a set value
    pub fn is_set(&self) -> bool {
        matches!(self, Self::Set(_))
    }

    /// Get a reference, if set
    pub fn as_ref(&self) -> Option<&E::ActiveModelEx> {
        match self {
            Self::Set(Some(model)) => Some(model),
            _ => None,
        }
    }

    /// Get a mutable reference, if set
    #[allow(clippy::should_implement_trait)]
    pub fn as_mut(&mut self) -> Option<&mut E::ActiveModelEx> {
        match self {
            Self::Set(Some(model)) => Some(model),
            _ => None,
        }
    }

    /// Return true if the containing model is set and changed
    pub fn is_changed(&self) -> bool {
        match self {
            Self::Set(Some(model)) => model.is_changed(),
            Self::Set(None) => true,
            Self::NotSet => false,
        }
    }

    /// Convert into an `Option<ActiveModelEx>`
    pub fn into_option(self) -> Option<E::ActiveModelEx> {
        match self {
            Self::Set(Some(model)) => Some(*model),
            Self::Set(None) | Self::NotSet => None,
        }
    }

    /// Convert this back to a `ModelEx` container
    pub fn try_into_model(self) -> Result<HasOne<E>, DbErr>
    where
        E::ActiveModelEx: TryIntoModel<E::ModelEx>,
    {
        Ok(match self {
            Self::Set(Some(model)) => HasOne::Loaded(Some(Box::new((*model).try_into_model()?))),
            Self::Set(None) => HasOne::Loaded(None),
            Self::NotSet => HasOne::Unloaded,
        })
    }
}

impl<E> ActiveHasMany<E>
where
    E: EntityTrait,
{
    /// Take ownership of the models, leaving `NotSet` in place
    pub fn take(&mut self) -> Self {
        std::mem::take(self)
    }

    /// Borrow models as slice
    /// NOTE: For [Self::Mutate] this will only return models that
    /// will be saved.
    pub fn as_slice(&self) -> &[E::ActiveModelEx] {
        match self {
            Self::Replace(models) | Self::Append(models) => models,
            Self::Mutate(mutations) => &mutations.save,
            Self::NotSet => &[],
        }
    }

    /// Iterate over all models being mutated
    pub fn iter(&self) -> ActiveHasManyBorrowedIter<'_, E> {
        match self {
            Self::Replace(models) | Self::Append(models) => {
                ActiveHasManyBorrowedIter::Slice(models.iter())
            }
            Self::Mutate(mutations) => ActiveHasManyBorrowedIter::Chained(
                mutations.save.iter().chain(mutations.delete.iter()),
            ),
            Self::NotSet => ActiveHasManyBorrowedIter::Empty(std::iter::empty()),
        }
    }

    /// Get a mutable vec. If self is `NotSet`, convert to append.
    /// NOTE: For [Self::Mutate] this will only return models that
    /// will be saved.
    pub fn as_mut_vec(&mut self) -> &mut Vec<E::ActiveModelEx> {
        match self {
            Self::Replace(models) | Self::Append(models) => models,
            Self::Mutate(mutations) => mutations.save_as_mut_vec(),
            Self::NotSet => {
                *self = Self::Append(vec![]);
                self.as_mut_vec()
            }
        }
    }

    /// Consume self as vector
    /// NOTE: For [Self::Mutate] this will only return models that
    /// will be saved.
    pub fn into_vec(self) -> Vec<E::ActiveModelEx> {
        match self {
            Self::Replace(models) | Self::Append(models) => models,
            Self::Mutate(mutations) => mutations.into_save_vec(),
            Self::NotSet => vec![],
        }
    }

    /// Returns an empty container of self
    pub fn empty_holder(&self) -> Self {
        match self {
            Self::Replace(_) => Self::Replace(vec![]),
            Self::Append(_) => Self::Append(vec![]),
            Self::Mutate(_) => Self::Mutate(Mutation::new()),
            Self::NotSet => Self::NotSet,
        }
    }

    /// Push an item to self
    ///
    /// If [Self::Mutate] and the given model matches a loaded model
    /// marked for deletion that model will not longer be deleted.
    pub fn push<AM: Into<E::ActiveModelEx>>(&mut self, model: AM) -> &mut Self {
        let model = model.into();
        match self {
            Self::Replace(models) | Self::Append(models) => models.push(model),
            Self::Mutate(mutations) => {
                let pk = model.get_primary_key_value();
                if pk.is_some() {
                    let idx = mutations
                        .delete
                        .iter()
                        .position(|x| x.get_primary_key_value() == pk);

                    if let Some(idx) = idx {
                        mutations.delete.swap_remove(idx);
                    }
                }
                mutations.save.push(model)
            }
            Self::NotSet => {
                *self = Self::Append(vec![model]);
            }
        }

        self
    }

    /// Add an ActiveModelEx that will be deleted on save. No changes occur
    /// if the primary key of the given model is not set.
    /// NOTE: Will promote [Self::Replace], [Self::Append], & [Self::NotSet] to [Self::Mutate]
    pub fn push_delete<AM: Into<E::ActiveModelEx>>(&mut self, model: AM) -> &mut Self {
        let model = model.into();

        let pk = model.get_primary_key_value();

        if pk.is_none() {
            return self;
        }

        match self {
            Self::Mutate(mutations) => {
                let idx = mutations
                    .save
                    .iter()
                    .position(|x| x.get_primary_key_value() == pk);

                if let Some(idx) = idx {
                    mutations.save.swap_remove(idx);
                }
                mutations.delete.push(model)
            }
            Self::Replace(models) | Self::Append(models) => {
                let mut models = std::mem::take(models);
                let idx = models.iter().position(|x| x.get_primary_key_value() == pk);

                if let Some(idx) = idx {
                    models.swap_remove(idx);
                }
                *self = Self::Mutate(Mutation::from_save_and_delete(models, vec![model]));
            }
            Self::NotSet => {
                *self = Self::Mutate(Mutation::from_delete(vec![model]));
            }
        }

        self
    }

    /// Mark a model loaded into the active model as deleted, if an active model
    /// with the given primary key is not loaded no action will be taken.
    /// NOTE: Will promote [Self::Replace], [Self::Append], & [Self::NotSet] to [Self::Mutate] even
    /// if no active model is found
    pub fn push_delete_existing(&mut self, pk: &ValueTuple) -> &mut Self {
        match self {
            Self::Mutate(mutations) => {
                let some_pk = Some(pk);
                let idx = mutations
                    .save
                    .iter()
                    .position(|x| x.get_primary_key_value().as_ref() == some_pk);

                if let Some(idx) = idx {
                    mutations.delete.push(mutations.save.swap_remove(idx));
                }
            }
            Self::Replace(models) | Self::Append(models) => {
                let mut models = std::mem::take(models);
                let some_pk = Some(pk);
                let idx = models
                    .iter()
                    .position(|x| x.get_primary_key_value().as_ref() == some_pk);

                if let Some(idx) = idx {
                    let model = models.swap_remove(idx);
                    *self = Self::Mutate(Mutation::from_save_and_delete(models, vec![model]));
                } else {
                    *self = Self::Mutate(Mutation::from_save(models));
                }
            }
            Self::NotSet => {
                *self = Self::Mutate(Mutation::new());
            }
        }

        self
    }

    /// Push an item to self, but convert Replace to Append
    pub fn append<AM: Into<E::ActiveModelEx>>(&mut self, model: AM) -> &mut Self {
        self.convert_to_append().push(model)
    }

    /// Replace all items in this set
    pub fn replace_all<I>(&mut self, models: I) -> &mut Self
    where
        I: IntoIterator<Item = E::ActiveModelEx>,
    {
        *self = Self::Replace(models.into_iter().collect());
        self
    }

    /// Convert [self] to [Self::Append], if set
    /// NOTE: If converting [self::Mutation] only models
    /// that will be saved/inserted will be retained
    pub fn convert_to_append(&mut self) -> &mut Self {
        match self.take() {
            Self::Replace(models) | Self::Append(models) => {
                *self = Self::Append(models);
            }
            Self::Mutate(mutations) => {
                *self = Self::Append(mutations.save);
            }
            Self::NotSet => {
                *self = Self::NotSet;
            }
        }

        self
    }

    /// Convert [self] to [Self::Mutate], if set
    pub fn convert_to_mutation(&mut self) -> &mut Self {
        match self.take() {
            Self::Replace(models) | Self::Append(models) => {
                *self = Self::Mutate(Mutation::from_save(models));
            }
            Self::Mutate(mutations) => {
                *self = Self::Mutate(mutations);
            }
            Self::NotSet => {
                *self = Self::NotSet;
            }
        }

        self
    }

    /// Reset self to NotSet
    pub fn not_set(&mut self) {
        *self = Self::NotSet;
    }

    /// If self is `Replace`
    pub fn is_replace(&self) -> bool {
        matches!(self, Self::Replace(_))
    }

    /// If self is `Append`
    pub fn is_append(&self) -> bool {
        matches!(self, Self::Append(_))
    }

    /// If self is `Mutate`
    pub fn is_mutate(&self) -> bool {
        matches!(self, Self::Mutate(_))
    }

    /// Return true if self is `Replace` or any containing model is changed
    pub fn is_changed(&self) -> bool {
        match self {
            Self::Replace(_) => true,
            Self::Append(models) => models.iter().any(|model| model.is_changed()),
            Self::Mutate(mutations) => {
                !mutations.delete.is_empty()
                    || mutations.save.iter().any(|model| model.is_changed())
            }
            Self::NotSet => false,
        }
    }

    /// Find within the models by primary key, return true if found
    pub fn find(&self, model: &E::Model) -> bool {
        let pk = model.get_primary_key_value();

        for item in self.as_slice() {
            if let Some(pk_item) = item.get_primary_key_value()
                && pk_item == pk
            {
                return true;
            }
        }

        false
    }

    /// Convert this back to a `ModelEx` container
    pub fn try_into_model(self) -> Result<HasMany<E>, DbErr>
    where
        E::ActiveModelEx: TryIntoModel<E::ModelEx>,
    {
        Ok(match self {
            Self::Replace(models) | Self::Append(models) => HasMany::Loaded(
                models
                    .into_iter()
                    .map(|t| t.try_into_model())
                    .collect::<Result<Vec<_>, DbErr>>()?,
            ),
            Self::Mutate(mutations) => HasMany::Loaded(
                mutations
                    .into_iter()
                    .map(|t| t.try_into_model())
                    .collect::<Result<Vec<_>, DbErr>>()?,
            ),
            Self::NotSet => HasMany::Unloaded,
        })
    }
}

impl<E: EntityTrait> From<ActiveHasMany<E>> for Option<Vec<E::ActiveModelEx>> {
    fn from(value: ActiveHasMany<E>) -> Self {
        match value {
            ActiveHasMany::NotSet => None,
            ActiveHasMany::Replace(models) | ActiveHasMany::Append(models) => Some(models),
            ActiveHasMany::Mutate(mutations) => Some(mutations.into_vec()),
        }
    }
}

impl<E: EntityTrait> Index<usize> for ActiveHasMany<E> {
    type Output = E::ActiveModelEx;

    fn index(&self, index: usize) -> &Self::Output {
        match self {
            ActiveHasMany::NotSet => {
                panic!("index out of bounds: the ActiveHasMany is NotSet (index: {index})")
            }
            ActiveHasMany::Replace(models) | ActiveHasMany::Append(models) => models.index(index),
            ActiveHasMany::Mutate(mutations) => mutations.save.index(index),
        }
    }
}

impl<E: EntityTrait> IndexMut<usize> for ActiveHasMany<E> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match self {
            ActiveHasMany::NotSet => {
                panic!("index out of bounds: the ActiveHasMany is NotSet (index: {index})")
            }
            ActiveHasMany::Replace(models) | ActiveHasMany::Append(models) => {
                models.index_mut(index)
            }
            ActiveHasMany::Mutate(mutations) => mutations.save.index_mut(index),
        }
    }
}

impl<E: EntityTrait> IntoIterator for ActiveHasMany<E> {
    type Item = E::ActiveModelEx;
    type IntoIter = std::vec::IntoIter<E::ActiveModelEx>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            ActiveHasMany::Replace(models) | ActiveHasMany::Append(models) => models.into_iter(),
            ActiveHasMany::Mutate(mutations) => mutations.save.into_iter(),
            ActiveHasMany::NotSet => Vec::new().into_iter(),
        }
    }
}

/// Converts from a set of models into `Append`, which performs non-destructive action
impl<E: EntityTrait> From<Vec<E::ActiveModelEx>> for ActiveHasMany<E> {
    fn from(value: Vec<E::ActiveModelEx>) -> Self {
        ActiveHasMany::Append(value)
    }
}
