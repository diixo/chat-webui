from __future__ import annotations
from typing import TYPE_CHECKING, Any

from loguru import logger
from datetime import datetime

from .rest_client import RestClient
from .user import User
from .tracker import Tracker
from .utils import loadable

class Project:
	"""Represents a project in codeBeamer."""

	if TYPE_CHECKING:
		_description: str | None
		_description_format: str | None
		_version: int | None
		_key_name: str | None
		_category: str | None
		_closed: bool | None
		_deleted: bool | None
		_template: bool | None
		_created_at: datetime | None
		_created_by: User | None
		_modified_at: datetime | None
		_modified_by: User | None
		_trackers: list[Tracker]
		_loaded: bool

	def __init__(self, id: int, name: str, **kwargs):
		self._id: int = id
		self._name: str = name
		self._client: RestClient = kwargs.get('client')
		self._trackers: list[Tracker] = list()
		self._loaded: bool = False
		# type only appears in GET /projects
		if not kwargs.get('type'):
			# if type is present then no other information is present
			self._load(kwargs)
		else:
			prop_defaults = {k: None for k in self.__class__.__annotations__}
			self.__dict__.update(prop_defaults)

	@property
	def id(self) -> int:
		"""The ID of the project."""
		return self._id

	@property
	def name(self) -> str:
		"""The name of the project."""
		return self._name

	@property
	@loadable
	def description(self) -> str | None:
		"""The project description."""
		return self._description

	@property
	@loadable
	def description_format(self) -> str | None:
		"""The format of the project description."""
		return self._description_format

	@property
	@loadable
	def version(self) -> int | None:
		"""The project version."""
		return self._version

	@property
	@loadable
	def key_name(self) -> str | None:
		"""The project key identifier."""
		return self._key_name

	@property
	@loadable
	def category(self) -> str | None:
		"""The category of the project."""
		return self._category

	@property
	@loadable
	def closed(self) -> bool | None:
		"""Flag for whether the project is closed or not."""
		return self._closed

	@property
	@loadable
	def deleted(self) -> bool | None:
		"""Flag for whether the project is deleted or not."""
		return self._deleted

	@property
	@loadable
	def template(self) -> bool | None:
		"""Flag for whether the project is a template or not."""
		return self._template

	@property
	@loadable
	def created_at(self) -> datetime | None:
		"""The datetime the project was created at."""
		return self._created_at

	@property
	@loadable
	def created_by(self) -> User | None:
		"""The user that created the project."""
		return self._created_by

	@property
	@loadable
	def modified_at(self) -> datetime | None:
		"""The datetime the project was last modified."""
		return self._modified_at

	@property
	@loadable
	def modified_by(self) -> User | None:
		"""The user that last modified the project."""
		return self._modified_by

	def _load(self, data: dict[str, Any] = None):
		"""Loads the rest of the project's data. When a project is fetched using 
		`Codebeamer.get_projects` only the ID and Name of the project are retrieved. 
		This prevents a lot of extra data that's not needed from being sent. Thus, 
		this method exists to flush out the rest of the project information if it is 
		needed."""
		if self._loaded:
			logger.info('Project already loaded, ignoring...')
			return
		if not data:
			data: dict[str, Any] = self._client.get(f'projects/{self.id}')
		logger.debug(data)
		self._description = data.get('description')
		self._description_format = data.get('descriptionFormat')
		self._version = data.get('version')
		self._key_name = data.get('keyName')
		self._category = data.get('category')
		self._closed = data.get('closed')
		self._deleted = data.get('deleted')
		self._template = data.get('template')
		self._created_at = datetime.strptime(data.get('createdAt'), '%Y-%m-%dT%H:%M:%S.%f')
		self._created_by = User(**data.get('createdBy'), client=self._client)
		self._modified_at = datetime.strptime(data.get('modifiedAt'), '%Y-%m-%dT%H:%M:%S.%f')
		self._modified_by = User(**data.get('modifiedBy'), client=self._client)
		self._trackers = list()
		self._loaded = True

	def get_trackers(self) -> list[Tracker]:
		"""Fetches all the trackers in this project.
		
		Returns:
		list[`Tracker`] — All the trackers under this project."""
		refs_trackers = self._client.get(f'projects/{self.id}/trackers')
		return [Tracker(**t, client=self._client, project=self) for t in refs_trackers]

	def get_wikipages(self) -> list[dict[str, Any]]:
		"""Fetches all wiki pages in this project.

		Returns:
		list[dict[str, Any]] — All wiki pages under this project.
		"""
		return self._client.get(f'projects/{self.id}/wikipages')

	def get_tracker(self, tracker: str | int) -> Tracker | None:
		"""Fetches a specific tracker from the project.
		
		Params:
		tracker — The name or ID of the tracker to fetch. — str | int
		
		Raises:
		TypeError — A type other than str or int was provided.
		
		Returns:
		`Tracker` — The tracker if it exists under the project."""
		if not self._trackers:
			self._trackers = self.get_trackers()
		if isinstance(tracker, int):
			trackers = {t.id: t for t in self._trackers}
		elif isinstance(tracker, str):
			trackers = {t.name: t for t in self._trackers}
		else:
			raise TypeError(f'expected str or int, got {type(tracker)}')
		return trackers.get(tracker)


	def __repr__(self) -> str:
		return f'Project(id={self.id}, name={self.name})'
	
	def __str__(self) -> str:
		return self.name
	
	def __eq__(self, o: object) -> bool:
		return isinstance(o, Project) and self.id == o.id
	
	def __lt__(self, o: object) -> bool:
		return isinstance(o, Project) and self.id < o.id


	def create_tracker(
		self,
		name: str,
		description: str,
		issue_type_id: int,
		user: User,
		key_name: str | None = None
	) -> Tracker:
		"""Create a tracker in the current project.

		Uses `TrackerModel` payload shape expected by v3-style
		`/projects/{projectId}/trackers` endpoint implementations.
		"""
		if not name:
			raise ValueError("*name is required to create a tracker")
		if not description:
			raise ValueError("*description is required to create a tracker")
		if issue_type_id is None:
			raise ValueError("*issue_type_id is required to create a tracker")
		if user is None:
			raise ValueError("*user is required to create a tracker")

		tracker_types = self._client.get('trackers/types')

		type_name = None
		for t in tracker_types:
			if t.get('id') == issue_type_id:
				type_name = t.get('name')
				break
		if type_name is None:
			return None

		payload = {
			"id": 0,	# stub, actual id is assigned by server

			"name": name,
			"keyName": key_name,
			"description": description,
			"hidden": False,

			"usingWorkflow": True,
			"usingQuickTransitions": False,

			"defaultShowAncestorItems": False,
			"defaultShowDescendantItems": False,
			"sharedInWorkingSet": True,
			"availableAsTemplate": False,
			"deleted": False,

			"type": {
				"id": issue_type_id,
				"name": type_name,
				"type": "TrackerTypeReference"
			},

			"project": {
				"id": self.id,
				"name": self.name,
				"type": "ProjectReference"
			},

			"createdBy": {
				"id": user.id,
				"name": user.name,
				"type": "UserReference",
				"email": user.email,
				"displayName": user.name,
			},

			"onlyWorkflowCanCreateNewReferringItem": False,
		}

		tracker_data = self._client.post(f"projects/{self.id}/trackers", json_=payload)

		if not isinstance(tracker_data, dict):
			raise ValueError("Failed to create tracker. Check request payload and permissions.")

		if tracker_data.get("message"):
			raise ValueError(f"Failed to create tracker: {tracker_data['message']}")

		if tracker_data.get("id", 0) > 0 and tracker_data.get("name") is not None:
			tracker_data["client"] = self._client
			return Tracker(**tracker_data)

		return None
